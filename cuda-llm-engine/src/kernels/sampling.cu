// Token 采样，单 block kernel：
//   1. 温度缩放（在 float 副本上操作）
//   2. top-k 截断（k 轮整块求最大值）
//   3. softmax
//   4. top-p（nucleus）截断：依次取最大概率，直到累积概率达到 top_p
//   5. 通过整块 CDF 扫描和主机端均匀随机数 u 进行多项式采样
// temperature <= 0 时选择贪心 argmax 路径。

#include "cuda_llm/ops.h"

#include <math.h>

namespace cudallm {
namespace op {

namespace {

constexpr int kBlock = 1024;
constexpr int kMaxNucleusRounds = 4096;

__device__ float bmax(float v, float *smem) {
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  v = (lane < 32) ? smem[lane] : -INFINITY;
  if (wid == 0) {
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

__device__ float bsum(float v, float *smem) {
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  v = (lane < 32) ? smem[lane] : 0.0f;
  if (wid == 0) {
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

__device__ int bmin_int(int v, int *smem_i) {
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    int o = __shfl_down_sync(0xffffffffu, v, off);
    v = min(v, o);
  }
  if (lane == 0) smem_i[wid] = v;
  __syncthreads();
  v = (lane < 32) ? smem_i[lane] : 0x7fffffff;
  if (wid == 0) {
    for (int off = 16; off > 0; off >>= 1)
      v = min(v, __shfl_down_sync(0xffffffffu, v, off));
    if (lane == 0) smem_i[0] = v;
  }
  __syncthreads();
  return smem_i[0];
}

__global__ void greedy_argmax_kernel(const __half *logits, int vocab,
                                     int *result) {
  __shared__ float smem[32];
  __shared__ int smem_i[32];
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < vocab; i += kBlock)
    mx = fmaxf(mx, __half2float(logits[i]));
  mx = bmax(mx, smem);

  int chosen = 0x7fffffff;
  for (int i = threadIdx.x; i < vocab; i += kBlock) {
    if (__half2float(logits[i]) == mx) chosen = min(chosen, i);
  }
  chosen = bmin_int(chosen, smem_i);
  if (threadIdx.x == 0) result[0] = chosen;
}

__global__ void sample_kernel(const __half *__restrict__ logits,
                              float *__restrict__ work, int *result,
                              int vocab, float temperature, int top_k,
                              float top_p, float u) {
  __shared__ float smem[32];
  __shared__ int smem_i[32];
  __shared__ float warp_total[32];
  __shared__ float warp_base[32];

  // 1) 拷贝并做温度缩放。
  for (int i = threadIdx.x; i < vocab; i += kBlock) {
    work[i] = __half2float(logits[i]) / temperature;
  }
  __syncthreads();

  // 2) Top-k。
  if (top_k > 0 && top_k < vocab) {
    float threshold = -INFINITY;
    for (int round = 0; round < top_k; ++round) {
      float mx = -INFINITY;
      for (int i = threadIdx.x; i < vocab; i += kBlock)
        mx = fmaxf(mx, work[i]);
      mx = bmax(mx, smem);
      if (round < top_k - 1) {
        for (int i = threadIdx.x; i < vocab; i += kBlock)
          if (work[i] == mx) work[i] = -INFINITY;
      }
      threshold = mx;
      __syncthreads();
    }
    for (int i = threadIdx.x; i < vocab; i += kBlock)
      if (work[i] < threshold) work[i] = -INFINITY;
    __syncthreads();
  }

  // 3) Softmax。
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < vocab; i += kBlock)
    mx = fmaxf(mx, work[i]);
  mx = bmax(mx, smem);

  float sum = 0.0f;
  for (int i = threadIdx.x; i < vocab; i += kBlock) {
    float e = expf(work[i] - mx);
    sum += e;
    work[i] = e;
  }
  sum = bsum(sum, smem);
  for (int i = threadIdx.x; i < vocab; i += kBlock)
    work[i] /= sum;
  __syncthreads();

  // 4) Top-p。取最大概率；保留的 token 编码为 -(2+p)。
  if (top_p < 1.0f) {
    float cumulative = 0.0f;
    for (int round = 0; round < kMaxNucleusRounds; ++round) {
      float m = -INFINITY;
      for (int i = threadIdx.x; i < vocab; i += kBlock)
        if (work[i] > 0.0f) m = fmaxf(m, work[i]);
      m = bmax(m, smem);
      if (m <= 0.0f) break;

      float cnt = 0.0f;
      for (int i = threadIdx.x; i < vocab; i += kBlock)
        if (work[i] == m) cnt += 1.0f;
      cnt = bsum(cnt, smem);

      if (cumulative >= top_p - 1e-7f) {
        for (int i = threadIdx.x; i < vocab; i += kBlock)
          if (work[i] == m) work[i] = 0.0f;
      } else {
        cumulative += cnt * m;
        for (int i = threadIdx.x; i < vocab; i += kBlock)
          if (work[i] == m) work[i] = -(2.0f + m);
      }
      __syncthreads();
    }
    // 恢复保留的概率；未被编码的项全部移除。
    for (int i = threadIdx.x; i < vocab; i += kBlock) {
      work[i] = (work[i] < -1.5f) ? (-2.0f - work[i]) : 0.0f;
    }
    __syncthreads();

    float s = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += kBlock) s += work[i];
    s = bsum(s, smem);
    for (int i = threadIdx.x; i < vocab; i += kBlock)
      work[i] /= s;
    __syncthreads();
  }

  // 5) CDF 扫描 + 多项式采样。
  float local_total = 0.0f;
  for (int i = threadIdx.x; i < vocab; i += kBlock)
    local_total += work[i];

  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  float v = local_total;
  for (int off = 16; off > 0; off >>= 1) {
    float n = __shfl_up_sync(0xffffffffu, v, off);
    if (lane >= off) v += n;
  }
  const float lane_base = v - local_total;
  if (lane == 0) warp_total[wid] = v;
  __syncthreads();

  if (wid == 0) {
    float wv = (lane < 32) ? warp_total[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      float n = __shfl_up_sync(0xffffffffu, wv, off);
      if (lane >= off) wv += n;
    }
    warp_base[lane] = wv - ((lane < 32) ? warp_total[lane] : 0.0f);
  }
  __syncthreads();

  float cum = warp_base[wid] + lane_base;
  int chosen = 0x7fffffff;
  for (int i = threadIdx.x; i < vocab; i += kBlock) {
    cum += work[i];
    if (cum >= u) chosen = min(chosen, i);
  }
  chosen = bmin_int(chosen, smem_i);
  // 安全兜底：FP32 归一化误差可能使总概率略小于 u，此时回退到最后一个
  // 有效 token，避免返回 0x7fffffff 导致下游越界。
  if (chosen >= vocab) chosen = vocab - 1;
  if (threadIdx.x == 0) result[0] = chosen;
}

}  // namespace

int sample(const Tensor &logits, float temperature, int top_k, float top_p,
           float u, Tensor &scratch, cudaStream_t stream) {
  const int vocab = static_cast<int>(logits.numel());
  float *work = scratch.data_ptr<float>();
  int *result = reinterpret_cast<int *>(work + vocab);

  if (temperature <= 0.0f) {
    CUDA_LAUNCH(greedy_argmax_kernel, 1, kBlock, 0, stream,
                logits.data_ptr<__half>(), vocab, result);
  } else {
    CUDA_LAUNCH(sample_kernel, 1, kBlock, 0, stream,
                logits.data_ptr<__half>(), work, result, vocab, temperature,
                top_k, top_p, u);
  }

  int token = -1;
  CUDA_CHECK(cudaMemcpyAsync(&token, result, sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return token;
}

}  // namespace op
}  // namespace cudallm
