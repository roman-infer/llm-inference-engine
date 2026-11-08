# 05 · KV Cache 机制

> 目标：理解自回归推理为什么必须缓存 K/V，缓存如何布局与追加，以及
> MHA/MQA/GQA 对它的影响。

## 1. 为什么需要 KV Cache

没有 cache 时，每生成一个 token 都要对**全部历史 token** 重新计算注意力，
总计算量随长度平方/立方膨胀，且历史 K/V 被反复重算。

KV cache 把每个位置投影后的 K、V 保存下来：

- prefill：一次性算好 prompt 全部位置的 K/V 并写入；
- decode：只为新 token 计算 1 个 K/V 追加，其余直接复用。

于是每步的注意力成本从"重算整段"降为"算 1 个新位置 + 读取历史"。

## 2. 布局与追加

本引擎布局（每层独立）：

```
k_cache : [n_kv_heads, max_seq_len, head_dim]
v_cache : 同
```

- 多层时外层再加一维 `[n_layer, ...]`，层内用 slice 取 view。
- `kv_append(cache, src, pos0)`：src 为 `[seq, n_kv_heads, head_dim]`，
  拷到 cache 的 `[pos0, pos0+seq)`；prefill 传整段、decode 传单个。
- 新序列开始前 `reset_caches` 清零。

## 3. 注意力如何使用

- prefill（causal）：位置 s 只 attend 到 `[0, s]`。
- decode：唯一的 query 位置 `past_len`，attend 到 `[0, past_len]` 全部缓存。

scores kernel 对 `t > past_len + s` 的位置填 `-inf`，softmax 后权重为 0。

## 4. MHA / MQA / GQA

```
MHA : n_kv_heads = n_heads
GQA : n_kv_heads < n_heads（多个 Q 头共享一组 KV）
MQA : n_kv_heads = 1（所有 Q 头共享一组 KV）
```

- 映射关系：`kv_head = q_head * n_kv_heads / n_heads`。
- KV cache 体积与 `n_kv_heads` 成正比，GQA/MQA 显著降低显存与带宽压力，
  这是现代 LLM 普遍采用 GQA 的原因。

## 5. 当前实现的局限（decode 并行度）

decode 时 query 只有 1 个，attention 的并行主要沿 head_dim（一个 warp
即可覆盖），而对历史长度 t 的遍历是**串行**的。长上下文、batch=1 时
GPU 并行度不足。

规划中的升级：

1. **FlashDecoding**：沿 KV 维度 split-K，多个 block 各自处理一段历史，
   再对输出做归约，把长 KV 也并行起来。
2. **PagedAttention**：KV cache 按固定大小 block 分页，用 block table
   间接寻址，支持显存碎片复用与 continuous batching。
3. **KV cache 量化**（FP8/INT8），进一步降带宽。
4. **Prefix caching**：相同 prompt 前缀的 KV 跨请求复用。

## 6. 容量估算

单层单 token KV 字节数：

```
2(K和V) × n_kv_heads × head_dim × dtype_bytes
```

规划显存时再乘以层数与最大序列长度；这决定了给定显存档位下可支持的
最大上下文与并发数。

## 7. 动手实验

1. 打印 prefill 前后 cache 中非零行数，确认追加位置正确。
2. 把 GQA 的 n_kv_heads 从 12 改成 4/1，对比 KV cache 体积与数值。
3. 统计一次完整生成中"投影计算量"在有/无 cache 下的差异。
