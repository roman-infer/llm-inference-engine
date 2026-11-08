// -*- cuda -*-
// =============================================================================
// ssd_kernel.cu：Mamba2 SSD（Structured State Space Duality）分块内核。
//
// 对单个 (batch, head) 的单个 chunk（长度 c，head_dim P，state 维 N），计算：
//   1) 块内前缀累积  cum[i,n] = sum_{k<=i} log_dA[k,n]
//   2) 块内衰减矩阵  M[i,j,n] = exp(cum[i,n]-cum[j,n])   (j<=i)
//   3) 对偶"注意力"  MCB[i,j] = sum_n C[i,n] * M[i,j,n] * B[j,n]
//   4) 块内输出      Y_intra[i,p] = sum_{j<=i} MCB[i,j] * X[j,p]
//   5) 跨块输出      Y_inter[i,p] = sum_n C[i,n] * exp(cum[i,n]) * S[n,p]
//   6) 块内状态增量  S_intra[n,p] = sum_j exp(cum[c-1,n]-cum[j,n]) * B[j,n] * X[j,p]
//   7) 块末状态      S_out = exp(cum[c-1,n]) * S + S_intra
//
// 这正是 PyTorch 版 ssd_chunkwise（lhengine/layers/mamba2.py）的单块计算。
// chunk 之间的 S 有顺序依赖，因此由 host 端按 chunk 循环调用本内核；
// 块内的 O(c^2*N + c*P*N + N*P*c) 计算在 GPU 上并行。
//
// 输入张量在 host 端已 permute 为 (bH, ...) 连续布局：
//   log_dA, B, C : (bH, L, N)；X : (bH, L, P)；S : (bH, N, P)。
// 数值用 float32（与官方 mamba2 内核的累加精度一致）。
// =============================================================================
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SSD_THREADS 256

// 每个 block 处理一个 (bH) 的一个 chunk。动态共享内存布局见 host。
__global__ void ssd_chunk_kernel(
        const float *__restrict__ X,      // (bH,L,P)
        const float *__restrict__ log_dA, // (bH,L,N)
        const float *__restrict__ B,      // (bH,L,N)
        const float *__restrict__ C,      // (bH,L,N)
        const float *__restrict__ S_in,   // (bH,N,P)
        float *__restrict__ Y,            // (bH,L,P)
        float *__restrict__ S_out,        // (bH,N,P)
        int chunk, int P, int N,
        int chunk_start, int L) {
    int bh = blockIdx.x;
    int tid = threadIdx.x;

    // 动态共享内存（由 launch 第三参数指定字节数）。
    extern __shared__ float scratch[];
    // 动态共享内存划分：
    //   sCum  : c*N   块内 cumsum
    //   sX    : c*P   块内 X
    //   sMCB  : c*c   对偶矩阵
    //   sS    : N*P   当前状态
    float *sCum = scratch;
    float *sX = sCum + chunk * N;
    float *sMCB = sX + chunk * P;
    float *sS = sMCB + chunk * chunk;

    // ---- 阶段0：加载 X，计算 cum，加载 S ----
    // X -> sX（越界位置填 0）。
    for (int idx = tid; idx < chunk * P; idx += SSD_THREADS) {
        int i = idx / P, p = idx % P;
        int gpos = chunk_start + i;
        sX[idx] = (gpos < L) ? X[bh * L * P + gpos * P + p] : 0.0f;
    }
    // cum：块内对 log_dA 做前缀和。每线程负责若干 (i,n)，顺序累加。
    for (int n = tid; n < N; n += SSD_THREADS) {
        float acc = 0.0f;
        for (int i = 0; i < chunk; ++i) {
            int gpos = chunk_start + i;
            if (gpos < L) acc += log_dA[bh * L * N + gpos * N + n];
            sCum[i * N + n] = acc;
        }
    }
    // S -> sS。
    for (int idx = tid; idx < N * P; idx += SSD_THREADS) {
        sS[idx] = S_in[bh * N * P + idx];
    }
    __syncthreads();

    // ---- 阶段1：MCB[i,j] = sum_n C[i,n]*exp(cum[i]-cum[j])*B[j,n] (j<=i) ----
    for (int ij = tid; ij < chunk * chunk; ij += SSD_THREADS) {
        int i = ij / chunk, j = ij % chunk;
        float val = 0.0f;
        if (j <= i) {
            int gi = chunk_start + i, gj = chunk_start + j;
            for (int n = 0; n < N; ++n) {
                float ci = C[bh * L * N + gi * N + n];
                float bj = B[bh * L * N + gj * N + n];
                float m = expf(sCum[i * N + n] - sCum[j * N + n]);
                val += ci * m * bj;
            }
        }
        sMCB[ij] = val;
    }
    __syncthreads();

    // ---- 阶段2：Y_intra[i,p] = sum_{j<=i} MCB[i,j]*X[j,p] ----
    // ---- 阶段3：Y_inter[i,p] = sum_n C[i,n]*exp(cum[i,n])*S[n,p] ----
    for (int ip = tid; ip < chunk * P; ip += SSD_THREADS) {
        int i = ip / P, p = ip % P;
        int gi = chunk_start + i;
        if (gi >= L) continue;
        float yintra = 0.0f;
        for (int j = 0; j <= i; ++j) yintra += sMCB[i * chunk + j] * sX[j * P + p];
        float yinter = 0.0f;
        for (int n = 0; n < N; ++n) {
            float ci = C[bh * L * N + gi * N + n];
            yinter += ci * expf(sCum[i * N + n]) * sS[n * P + p];
        }
        Y[bh * L * P + gi * P + p] = yintra + yinter;
    }
    __syncthreads();

    // ---- 阶段4：S_intra[n,p] = sum_j exp(cum[c-1]-cum[j])*B[j,n]*X[j,p] ----
    // 取块内最后一个真实位置作为 c-1（padding 不影响，因为 cum 在 padding 处不再增长）。
    int last = chunk - 1;
    for (int np = tid; np < N * P; np += SSD_THREADS) {
        int n = np / P, p = np % P;
        float sintra = 0.0f;
        for (int j = 0; j < chunk; ++j) {
            int gj = chunk_start + j;
            if (gj >= L) break;
            float bj = B[bh * L * N + gj * N + n];
            float ed = expf(sCum[last * N + n] - sCum[j * N + n]);
            sintra += ed * bj * sX[j * P + p];
        }
        // 阶段5：S_out = exp(cum[last])*S + S_intra。
        float cd = (chunk_start + last < L) ? expf(sCum[last * N + n]) : 1.0f;
        S_out[bh * N * P + n * P + p] = cd * sS[n * P + p] + sintra;
    }
}

// Host 端：把输入 permute 成 (bH,...)，逐 chunk 调用内核，最后还原 S 布局。
std::tuple<torch::Tensor, torch::Tensor>
ssd_chunkwise_cuda(torch::Tensor X, torch::Tensor log_dA,
                   torch::Tensor B, torch::Tensor C,
                   torch::Tensor S, int chunk) {
    // 输入布局 (b,L,H,N/P)。permute 到 (b,H,L,N/P) 并展平 bH。
    auto Xp = X.permute({0, 2, 1, 3}).contiguous();       // (b,H,L,P)
    auto Ap = log_dA.permute({0, 2, 1, 3}).contiguous();  // (b,H,L,N)
    auto Bp = B.permute({0, 2, 1, 3}).contiguous();
    auto Cp = C.permute({0, 2, 1, 3}).contiguous();
    int64_t b = X.size(0), H = X.size(2), L = X.size(1);
    int P = X.size(3), N = log_dA.size(3);
    int64_t bH = b * H;

    auto Xf = Xp.reshape({bH, L, P});
    auto Af = Ap.reshape({bH, L, N});
    auto Bf = Bp.reshape({bH, L, N});
    auto Cf = Cp.reshape({bH, L, N});
    // S：(b,H,N,P) -> (bH,N,P)
    auto Sf = S.reshape({b, H, N, P}).reshape({bH, N, P}).contiguous();

    auto Yf = torch::empty_like(Xf);
    auto Srun = Sf;
    int n_chunks = (static_cast<int>(L) + chunk - 1) / chunk;

    // 动态共享内存：c*N + c*P + c*c + N*P。
    size_t smem = (chunk * N + chunk * P + chunk * chunk + N * P) * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();
    for (int q = 0; q < n_chunks; ++q) {
        auto Snew = torch::empty_like(Srun);
        dim3 grid(bH);
        dim3 block(SSD_THREADS);
        // 为该 kernel 申请更大的动态共享内存（首次设置属性即可）。
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(ssd_chunk_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem));
            configured = true;
        }
        ssd_chunk_kernel<<<grid, block, smem, stream>>>(
            Xf.data_ptr<float>(), Af.data_ptr<float>(),
            Bf.data_ptr<float>(), Cf.data_ptr<float>(),
            Srun.data_ptr<float>(), Yf.data_ptr<float>(),
            Snew.data_ptr<float>(),
            chunk, P, N, q * chunk, static_cast<int>(L));
        Srun = Snew;
    }

    // 还原输出布局：Y (bH,L,P) -> (b,H,L,P) -> (b,L,H,P)。
    auto Y = Yf.reshape({b, H, L, P}).permute({0, 2, 1, 3}).contiguous();
    auto Sout = Srun.reshape({b, H, N, P});
    return std::make_tuple(Y, Sout);
}
