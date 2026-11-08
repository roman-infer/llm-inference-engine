#pragma once

#include <cstdint>

#include "cuda_llm/cuda_utils.h"  // CUDA_CHECK / CUDA_LAUNCH
#include "cuda_llm/gemm.h"
#include "cuda_llm/tensor.h"

namespace cudallm {
namespace op {

// ---------------------------------------------------------------------------
// 归一化层。x/y 形状为 [..., cols]；weight（和 bias）形状为 [cols]。
// ---------------------------------------------------------------------------
void rms_norm(const Tensor &x, const Tensor &weight, Tensor &y, float eps,
              cudaStream_t stream);
void layer_norm(const Tensor &x, const Tensor &weight, const Tensor *bias,
                Tensor &y, float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 线性层。
//   wt : [in,out]（已转置，GEMM 友好）-> y = x * wt
//   w  : [out,in]（GEMV 友好，decode 用）-> y = x * w^T
// bias 可选，形状 [out]。
// ---------------------------------------------------------------------------
struct LinearParams {
  const Tensor *wt = nullptr;
  const Tensor *w = nullptr;
  const Tensor *bias = nullptr;
};

void linear_gemm(const Tensor &x, const LinearParams &lp, Tensor &y,
                 GemmVersion version, cudaStream_t stream);
void linear_gemv(const Tensor &x, const LinearParams &lp, Tensor &y,
                 cudaStream_t stream);

// 二维转置：src [r,c] -> dst [c,r]。在模型构建时调用一次，
// 用于将权重预打包为 GEMM 布局。
void transpose_2d(const Tensor &src, Tensor &dst, cudaStream_t stream);

// 把 [rows, 3*cols] 的 qkv 投影结果按列切成三段连续缓冲（各 [rows, cols]）。
// 说明：Tensor 不带 stride，切片后的 view 会按连续布局误读，
// 因此这里显式拷贝，而不是 slice(dim=1) + view。
void split_qkv(const Tensor &src, Tensor &q, Tensor &k, Tensor &v,
               cudaStream_t stream);

// ---------------------------------------------------------------------------
// 旋转位置编码（RoPE）。
// q : [seq, n_heads,    head_dim]
// k : [seq, n_kv_heads, head_dim]
// 位置范围为 [pos0, pos0+seq)；原地修改。
// ---------------------------------------------------------------------------
void rope_inplace(Tensor &q, Tensor &k, int pos0, float theta_base,
                  cudaStream_t stream);

// ---------------------------------------------------------------------------
// KV cache。
// cache : [n_kv_heads, max_seq, head_dim]
// src   : [seq, n_kv_heads, head_dim]
// 将 src 拷贝到 cache 中位置 [pos0, pos0+seq) 处。
// ---------------------------------------------------------------------------
void kv_append(Tensor &cache, const Tensor &src, int pos0,
               cudaStream_t stream);

// ---------------------------------------------------------------------------
// 多头注意力（通过 n_kv_heads 支持 MHA / MQA / GQA）。
// q       : [seq, n_heads, d]
// k_cache : [n_kv_heads, max_seq, d]   （v 布局相同）
// out     : [seq, n_heads, d]
// 对 kv 位置 [0, past_len+seq) 做注意力；因果掩码屏蔽未来位置。
// scratch 提供设备端工作区（seq*kv_len 个 half）。
// ---------------------------------------------------------------------------
void attention(const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
               Tensor &out, int n_heads, int n_kv_heads, int past_len,
               bool causal, Tensor &scratch, cudaStream_t stream);

// 在线 softmax 版本（单趟）：一个 warp 处理一个 (s,h)，沿 kv 流式更新
// (m, l, o)，不物化 [n_heads*seq, kv_len] 分数矩阵。
// 显存 O(seq*d)；无需 scratch。与 attention() 数值等价（fp16 累加差异内）。
void attention_online(const Tensor &q, const Tensor &k_cache,
                      const Tensor &v_cache, Tensor &out, int n_heads,
                      int n_kv_heads, int past_len, bool causal,
                      cudaStream_t stream);

// ---------------------------------------------------------------------------
// 激活函数 / 逐元素操作（仅支持形状完全相同的广播）。
// ---------------------------------------------------------------------------
void silu(const Tensor &x, Tensor &y, cudaStream_t stream);
void gelu(const Tensor &x, Tensor &y, cudaStream_t stream);

void add_inplace(Tensor &a, const Tensor &b, cudaStream_t stream);
void add(const Tensor &a, const Tensor &b, Tensor &y, cudaStream_t stream);
void mul(const Tensor &a, const Tensor &b, Tensor &y, cudaStream_t stream);

// 对 [rows, cols] 逐行 softmax，原地操作。
void softmax(Tensor &x, int rows, int cols, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 嵌入层查表：table [vocab, dim]，ids [seq]（int32），out [seq, dim]。
// ---------------------------------------------------------------------------
void embedding(const Tensor &table, const Tensor &ids, Tensor &out,
               cudaStream_t stream);

// ---------------------------------------------------------------------------
// 精度转换。
// ---------------------------------------------------------------------------
void cast_f32_to_f16(const float *src, __half *dst, int64_t n,
                     cudaStream_t stream);
void cast_f16_to_f32(const __half *src, float *dst, int64_t n,
                     cudaStream_t stream);

// ---------------------------------------------------------------------------
// 采样。logits [vocab]；u 为 [0,1) 区间内主机端生成的均匀随机数。
// temperature <= 0 -> 贪心 argmax。否则进行温度缩放、top-k 和 top-p
// 截断，然后多项式分布采样。scratch 提供工作区。
// 返回选中的 token id。
// ---------------------------------------------------------------------------
int sample(const Tensor &logits, float temperature, int top_k, float top_p,
           float u, Tensor &scratch, cudaStream_t stream);

}  // namespace op
}  // namespace cudallm
