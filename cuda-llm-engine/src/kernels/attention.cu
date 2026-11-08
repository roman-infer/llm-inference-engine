// 多头注意力。提供两条数值等价、代价模型不同的路径：
//
//   attention()        三趟参考实现（scores -> softmax -> weighted sum），
//                      可读性最好，但会在显存里物化 [n_heads*seq, kv_len]
//                      的分数矩阵，显存 O(S*KV)。
//   attention_online() 在线 softmax（FlashAttention 的流式写法），一个 warp
//                      负责一个 (s, h)，沿 kv 顺序流式更新 m / l / o，
//                      不物化分数矩阵，显存 O(S*d)，代价是沿 kv 串行、
//                      单行并行度只有 d 而非 kv。
//
// 两条路径都按 1/sqrt(d) 缩放分数（缩放放在 kernel 里，调用方无需预处理 q）。
// MHA / MQA / GQA 的区别仅在于 n_kv_heads，kv head 索引为
//   kh = h * n_kv_heads / n_heads。

#include <cmath>

#include "cuda_llm/ops.h"

namespace cudallm {
namespace op {

namespace {

// ---------------------------------------------------------------------------
// 三趟参考实现
// ---------------------------------------------------------------------------

// 第 1 趟：每个 (s,h) 一个 block，4 个 warp 跨步遍历 t；lane 归约 d 维。
__global__ void attention_scores_kernel(
    const __half *__restrict__ q, const __half *__restrict__ kc,
    __half *__restrict__ scores, int seq, int n_heads, int n_kv_heads,
    int kv_len, int head_dim, int max_seq, int past_len, bool causal,
    float scale) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int kh = h * n_kv_heads / n_heads;

  extern __shared__ char raw[];
  __half *qrow = reinterpret_cast<__half *>(raw);
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    qrow[i] = q[(static_cast<int64_t>(s) * n_heads + h) * head_dim + i];
  }
  __syncthreads();

  const int n_warps = blockDim.x / 32;
  const int wid = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  for (int t = wid; t < kv_len; t += n_warps) {
    float dot = 0.0f;
    const __half *kr =
        kc + (static_cast<int64_t>(kh) * max_seq + t) * head_dim;
    for (int i = lane; i < head_dim; i += 32) {
      dot += __half2float(qrow[i]) * __half2float(kr[i]);
    }
    for (int off = 16; off > 0; off >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, off);
    }
    if (lane == 0) {
      float v = dot * scale;
      if (causal && t > past_len + s) v = -INFINITY;
      scores[(static_cast<int64_t>(h) * seq + s) * kv_len + t] =
          __float2half(v);
    }
  }
}

// 第 3 趟：每个 (s,h) 一个 warp；lane 持有 head_dim 个累加器，t 流式读取。
__global__ void attention_weighted_kernel(
    const __half *__restrict__ scores, const __half *__restrict__ vc,
    __half *__restrict__ out, int seq, int n_heads, int n_kv_heads,
    int kv_len, int head_dim, int max_seq) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int kh = h * n_kv_heads / n_heads;
  const int lane = threadIdx.x;

  const __half *srow =
      scores + (static_cast<int64_t>(h) * seq + s) * kv_len;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int t = 0; t < kv_len; ++t) {
    const float sc = __half2float(srow[t]);
    const __half *vr =
        vc + (static_cast<int64_t>(kh) * max_seq + t) * head_dim;
    if (lane < head_dim) acc0 += sc * __half2float(vr[lane]);
    if (head_dim > 32 && lane + 32 < head_dim)
      acc1 += sc * __half2float(vr[lane + 32]);
  }

  __half *op =
      &out[(static_cast<int64_t>(s) * n_heads + h) * head_dim];
  if (lane < head_dim) op[lane] = __float2half(acc0);
  if (head_dim > 32 && lane + 32 < head_dim)
    op[lane + 32] = __float2half(acc1);
}

// ---------------------------------------------------------------------------
// 在线 softmax（单趟）
// ---------------------------------------------------------------------------

constexpr int kWarpsPerBlock = 4;
constexpr int kMaxDPerLane = 4;  // 支持 head_dim <= 128

__device__ __forceinline__ float warp_all_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

// 一个 warp 处理一个 (s,h)：沿 t 流式读 k/v，在线更新 (m, l, o)。
// 每个 lane 持有 head_dim 中步长 32 的那几个分量。
__global__ void attention_online_kernel(
    const __half *__restrict__ q, const __half *__restrict__ kc,
    const __half *__restrict__ vc, __half *__restrict__ out, int seq,
    int n_heads, int n_kv_heads, int kv_len, int head_dim, int max_seq,
    int past_len, bool causal, float scale) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= seq * n_heads) return;
  if (lane >= head_dim) return;

  const int s = row / n_heads;
  const int h = row - s * n_heads;
  const int kh = h * n_kv_heads / n_heads;

  // q 的一行拆到 lane 上：lane 持有 d = lane, lane+32, ...
  const __half *qp = q + (static_cast<int64_t>(s) * n_heads + h) * head_dim;
  float qv[kMaxDPerLane];
#pragma unroll
  for (int i = 0; i < kMaxDPerLane; ++i) {
    const int d = lane + i * 32;
    qv[i] = (d < head_dim) ? __half2float(qp[d]) : 0.0f;
  }

  float m = -INFINITY;  // 行最大值
  float l = 0.0f;       // 行指数和
  float ov[kMaxDPerLane];
#pragma unroll
  for (int i = 0; i < kMaxDPerLane; ++i) ov[i] = 0.0f;

  const __half *kb =
      kc + static_cast<int64_t>(kh) * max_seq * head_dim;
  const __half *vb =
      vc + static_cast<int64_t>(kh) * max_seq * head_dim;

  // 因果掩码下第 s 行只看到 [0, past_len + s]，不需要遍历整段 kv。
  const int t_end = causal ? (past_len + s) : (kv_len - 1);

  for (int t = 0; t <= t_end; ++t) {
    const __half *kp = kb + static_cast<int64_t>(t) * head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxDPerLane; ++i) {
      const int d = lane + i * 32;
      if (d < head_dim) dot += qv[i] * __half2float(kp[d]);
    }
    dot = warp_all_sum(dot) * scale;

    const float m_new = fmaxf(m, dot);
    const float alpha = __expf(m - m_new);  // m = -inf 时为 0，首轮即正确
    const float p = __expf(dot - m_new);
    l = l * alpha + p;

    const __half *vp = vb + static_cast<int64_t>(t) * head_dim;
#pragma unroll
    for (int i = 0; i < kMaxDPerLane; ++i) {
      const int d = lane + i * 32;
      if (d < head_dim) {
        ov[i] = ov[i] * alpha + p * __half2float(vp[d]);
      }
    }
    m = m_new;
  }

  const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  __half *op = out + (static_cast<int64_t>(s) * n_heads + h) * head_dim;
#pragma unroll
  for (int i = 0; i < kMaxDPerLane; ++i) {
    const int d = lane + i * 32;
    if (d < head_dim) op[d] = __float2half(ov[i] * inv);
  }
}

}  // namespace

void attention(const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
               Tensor &out, int n_heads, int n_kv_heads, int past_len,
               bool causal, Tensor &scratch, cudaStream_t stream) {
  const int seq = static_cast<int>(q.size(0));
  const int head_dim = static_cast<int>(q.size(2));
  const int max_seq = static_cast<int>(k_cache.size(1));
  const int kv_len = past_len + seq;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  __half *sp = scratch.data_ptr<__half>();

  // 第 1 趟
  {
    dim3 grid(seq, n_heads);
    dim3 block(128);
    CUDA_LAUNCH(attention_scores_kernel, grid, block,
                head_dim * sizeof(__half), stream,
                q.data_ptr<__half>(), k_cache.data_ptr<__half>(), sp, seq,
                n_heads, n_kv_heads, kv_len, head_dim, max_seq, past_len,
                causal, scale);
  }

  // 第 2 趟：对 [n_heads*seq, kv_len] 做 softmax。
  {
    Tensor st = Tensor::wrap(
        sp,
        std::vector<int64_t>{static_cast<int64_t>(n_heads) * seq, kv_len},
        DType::kFloat16);
    softmax(st, n_heads * seq, kv_len, stream);
  }

  // 第 3 趟
  {
    dim3 grid(seq, n_heads);
    dim3 block(32);
    CUDA_LAUNCH(attention_weighted_kernel, grid, block, 0, stream, sp,
                v_cache.data_ptr<__half>(), out.data_ptr<__half>(), seq,
                n_heads, n_kv_heads, kv_len, head_dim, max_seq);
  }
}

void attention_online(const Tensor &q, const Tensor &k_cache,
                      const Tensor &v_cache, Tensor &out, int n_heads,
                      int n_kv_heads, int past_len, bool causal,
                      cudaStream_t stream) {
  const int seq = static_cast<int>(q.size(0));
  const int head_dim = static_cast<int>(q.size(2));
  const int max_seq = static_cast<int>(k_cache.size(1));
  const int kv_len = past_len + seq;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const int rows = seq * n_heads;
  const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  CUDA_LAUNCH(attention_online_kernel, blocks, kWarpsPerBlock * 32, 0,
              stream, q.data_ptr<__half>(), k_cache.data_ptr<__half>(),
              v_cache.data_ptr<__half>(), out.data_ptr<__half>(), seq,
              n_heads, n_kv_heads, kv_len, head_dim, max_seq, past_len,
              causal, scale);
}

}  // namespace op
}  // namespace cudallm
