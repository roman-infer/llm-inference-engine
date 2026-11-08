// -*- cuda -*-
// =============================================================================
// scan_kernel.cu：一阶时变递推的并行（分块）扫描 CUDA 内核。
//
// 要算的递推（序列在最后一维，长度 L）：
//      h_t = a_t * h_{t-1} + b_t,   h_{-1} = 0
// 其中 a_t、b_t 对每个通道、每个位置都可能不同（Mamba1 的"选择性"）。
//
// 把递推看成对"变换三元组"的关联运算。记单步变换为 (a, b)，含义：
//      输入 h  ->  输出 a*h + b
// 两个变换的复合（先经过 far、再经过 near）：
//      near(a_n, b_n) after far(a_f, b_f)
//      h -> a_n*(a_f*h + b_f) + b_n = (a_n*a_f)*h + (a_n*b_f + b_n)
// 即关联算子：
//      (a_n, b_n) ∘ (a_f, b_f) = (a_n*a_f, a_n*b_f + b_n)
// 因为该算子满足结合律，整个序列的"总变换"可以用并行前缀扫描求出来。
//
// 本文件采用经典的三段式（两 kernel + 一个 fixup）：
//   阶段1  block_scan_kernel：每个 block 处理一个通道的一个 chunk，
//          块内用 Hillis-Steele 并行扫描，写出"块内 partial"，
//          并把每个 chunk 的总 carry（a_total, b_total）写到 carry 缓冲；
//   阶段2  carry_scan_kernel：对每个通道的 chunk-carry 做顺序扫描，
//          得到每个 chunk 之前（左侧）所有 chunk 的前缀 carry；
//   阶段3  apply_carry_kernel：把前缀 carry 应用到块内 partial，
//          h = a_pre*h_partial + b_pre，得到最终结果。
//
// 说明：这是一个"清晰正确、便于讲解"的实现，刻意不做官方 selective_scan
// 那种极限融合（它把离散化、扫描、写回融在一个内核里）。数值上用 float 累加，
// 与官方 mamba-ssm 的精度策略一致。
// =============================================================================
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 每个 chunk 的线程数（也是 chunk 的长度）。Hillis-Steele 需要 log2(THREADS) 轮。
#define SCAN_THREADS 256

// 单个变换（a, b）的复合：near 作用在 far 之后。
__device__ __forceinline__ void combine(float &a_n, float &b_n,
                                        float a_f, float b_f) {
    // 注意：先更新 b（用到旧的 a_n），再更新 a，顺序不能反。
    b_n = a_n * b_f + b_n;
    a_n = a_n * a_f;
}

// -----------------------------------------------------------------------------
// 阶段1：块内并行扫描 + 记录 chunk carry。
// 输入  a, b : (C, L)，C 个通道、L 个序列位置（最后一维连续）。
// 输出  y    : (C, L)，块内 inclusive 扫描结果（尚未加跨块 carry）。
// 输出  ca, cb: (C, n_chunks)，每个 chunk 的总 carry。
// -----------------------------------------------------------------------------
__global__ void block_scan_kernel(const float *__restrict__ a,
                                  const float *__restrict__ b,
                                  float *__restrict__ y,
                                  float *__restrict__ ca,
                                  float *__restrict__ cb,
                                  int L, int n_chunks) {
    int c = blockIdx.y;          // 通道号
    int chunk = blockIdx.x;      // chunk 号
    int tid = threadIdx.x;       // 块内位置（0..SCAN_THREADS-1）
    int pos = chunk * SCAN_THREADS + tid;  // 全局序列位置

    // 共享内存：Hillis-Steele 过程中保存当前 (a, b)。
    __shared__ float sa[SCAN_THREADS];
    __shared__ float sb[SCAN_THREADS];

    // 越界（chunk 末尾的 padding 位置）当作"恒等变换"：a=1, b=0。
    float va = 1.0f, vb = 0.0f;
    if (pos < L) {
        int idx = c * L + pos;
        va = a[idx];
        vb = b[idx];
    }
    sa[tid] = va;
    sb[tid] = vb;
    __syncthreads();

    // Hillis-Steele inclusive 前缀扫描：offset = 1,2,4,...
    for (int offset = 1; offset < SCAN_THREADS; offset <<= 1) {
        float la, lb;
        if (tid >= offset) {
            // 取更早（tid-offset）处已经累积的变换。
            la = sa[tid - offset];
            lb = sb[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            // 当前（近）复合更早（远）。
            float na = sa[tid], nb = sb[tid];
            combine(na, nb, la, lb);
            sa[tid] = na;
            sb[tid] = nb;
        }
        __syncthreads();
    }

    // 写出块内 partial 结果（真实位置才写）。
    if (pos < L) {
        int idx = c * L + pos;
        y[idx] = sb[tid];   // inclusive 扫描的 b 分量即"从 0 起点的 h"
    }

    // chunk 的总 carry = 最后一个线程（覆盖该 chunk 最右位置）的变换。
    if (tid == SCAN_THREADS - 1) {
        int ci = c * n_chunks + chunk;
        ca[ci] = sa[tid];
        cb[ci] = sb[tid];
    }
}

// -----------------------------------------------------------------------------
// 阶段2：对每个通道的 chunk carry 做顺序扫描，得到每个 chunk 的"前缀 carry"。
// 前缀 carry 表示该 chunk 左侧所有 chunk 的复合变换。
// 输入/输出 ca, cb : (C, n_chunks)，原地改写为"排他前缀"（exclusive）。
// 用单线程逐 chunk 顺序累加（每通道一个线程），chunk 数通常较少。
// -----------------------------------------------------------------------------
__global__ void carry_scan_kernel(float *__restrict__ ca,
                                  float *__restrict__ cb,
                                  int n_chunks) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // 通道号
    // 前缀的初始值为恒等变换 (a=1, b=0)。
    float pre_a = 1.0f, pre_b = 0.0f;
    for (int ch = 0; ch < n_chunks; ++ch) {
        int idx = c * n_chunks + ch;
        float cur_a = ca[idx];
        float cur_b = cb[idx];
        // 先把"左侧前缀"写回（排他），供阶段3使用。
        ca[idx] = pre_a;
        cb[idx] = pre_b;
        // 再把当前 chunk 复合进前缀：前缀 = 当前 chunk after 旧前缀。
        float na = cur_a, nb = cur_b;
        combine(na, nb, pre_a, pre_b);
        pre_a = na;
        pre_b = nb;
    }
}

// -----------------------------------------------------------------------------
// 阶段3：把每个 chunk 的前缀 carry 应用到块内 partial 结果。
// 若 chunk 左侧前缀为 (a_pre, b_pre)，则最终
//      h = a_pre * h_partial + b_pre
// （块内 partial 是"假设该 chunk 从 0 起点"的结果；加前缀后等价于从真实历史出发。）
// -----------------------------------------------------------------------------
__global__ void apply_carry_kernel(float *__restrict__ y,
                                   const float *__restrict__ ca,
                                   const float *__restrict__ cb,
                                   int L, int n_chunks) {
    int c = blockIdx.y;
    int chunk = blockIdx.x;
    int tid = threadIdx.x;
    int pos = chunk * SCAN_THREADS + tid;
    if (pos < L) {
        int ci = c * n_chunks + chunk;
        float a_pre = ca[ci];
        float b_pre = cb[ci];
        int idx = c * L + pos;
        y[idx] = a_pre * y[idx] + b_pre;
    }
}

// -----------------------------------------------------------------------------
// Host 端入口：分配缓冲、依次 launch 三个 kernel。
// 输入 a, b : (C, L) contiguous float。
// 输出 y    : (C, L)。
// -----------------------------------------------------------------------------
torch::Tensor scan_forward_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "scan 输入必须在 CUDA 上");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "目前仅支持 float32 scan");
    TORCH_CHECK(a.sizes() == b.sizes(), "a 与 b 形状必须一致");

    int C = static_cast<int>(a.size(0));
    int L = static_cast<int>(a.size(1));
    int n_chunks = (L + SCAN_THREADS - 1) / SCAN_THREADS;

    auto y = torch::empty_like(b);
    auto opts = a.options();
    // chunk carry 缓冲：(C, n_chunks)。
    auto ca = torch::empty({C, n_chunks}, opts);
    auto cb = torch::empty({C, n_chunks}, opts);

    // 阶段1/3 的 grid：(n_chunks, C)，每 block SCAN_THREADS 线程。
    dim3 grid(n_chunks, C);
    dim3 block(SCAN_THREADS);
    block_scan_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                       y.data_ptr<float>(),
                                       ca.data_ptr<float>(), cb.data_ptr<float>(),
                                       L, n_chunks);

    // 阶段2：每通道一个线程，顺序扫描 chunk carry。
    int carry_threads = ((C + 255) / 256) * 256;
    carry_scan_kernel<<<1, carry_threads>>>(ca.data_ptr<float>(),
                                            cb.data_ptr<float>(), n_chunks);

    // 阶段3：应用前缀 carry。
    apply_carry_kernel<<<grid, block>>>(y.data_ptr<float>(),
                                        ca.data_ptr<float>(),
                                        cb.data_ptr<float>(),
                                        L, n_chunks);
    return y;
}
