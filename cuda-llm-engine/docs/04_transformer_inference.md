# 04 · Transformer 推理流程

> 目标：把一个 token 从"输入 id"走到"输出下一个 token 的概率"的全过程
> 讲清楚。本引擎实现了 GPT-2（learned PE + GELU）与 LLaMA（RoPE +
> SwiGLU）两种结构。

## 1. 两种 Block 对照

**GPT-2 block（pre-LN）**

```
x
├─ LayerNorm ─ QKV 投影 ─ MHA(causal) ─ 输出投影 ─┐
└────────────────────────────────────────────(+)─ x'
x'
├─ LayerNorm ─ fc ─ GELU ─ proj ───────────────┐
└──────────────────────────────────────────(+)─ x''
```

- 位置编码：可学习的 wpe，与 token embedding 相加。
- 输出头：与输入 embedding 共享权重（tied）。

**LLaMA block（pre-RMSNorm）**

```
x
├─ RMSNorm ─ Q/K/V 投影 ─ RoPE ─ GQA(causal) ─ O 投影 ─┐
└────────────────────────────────────────────────(+)─ x'
x'
├─ RMSNorm ─ gate、up 投影 ─ SiLU(gate)⊙up ─ down ─┐
└──────────────────────────────────────────────(+)─ x''
```

- 无 bias；最终 RMSNorm；lm_head 独立或 tied。

## 2. 算子清单（本引擎全部手写）

| 类别 | 算子 |
| --- | --- |
| 归一化 | LayerNorm、RMSNorm |
| 注意力 | MHA / MQA / GQA、RoPE、KV Cache append |
| 激活 | SiLU、GELU（exact） |
| 采样 | Temperature、Top-K、Top-P、多项式采样 / greedy |
| 其他 | Embedding、逐元素 add/mul、F32↔F16 转换、2D transpose |

## 3. Attention 的三个 pass（当前实现）

```
scores[h,s,t] = Q[s,h,:]·K[h,t,:]        （causal 位置填 -inf）
P  = softmax(scores, over t)
O[s,h,:] = Σ_t P[h,s,t] · V[h,t,:]
```

- 一个 block 处理一个 (s,h)：scores kernel 中 lane 沿 head_dim 归约；
  weighted kernel 中 lane 持有 head_dim 累加器、沿 t 流式累加。
- GQA/MQA 只改变 KV 头数：`kh = h * n_kv_heads / n_heads`。
- 这是**显式三 pass** 的可读版本，会物化 P 矩阵；在线 softmax
  （FlashAttention 风格）是规划中的升级。

## 4. Prefill vs Decode

| | Prefill | Decode |
| --- | --- | --- |
| 输入 | 整段 prompt（n 个 token） | 1 个 token |
| 形状 | GEMM，compute 可并行 | GEMV，带宽受限 |
| KV cache | 写入整段 | 每次追加 1 行 |
| 关注点 | 算力 / 融合 | launch 开销、带宽、cache 命中 |

生成流程（见 `generate.cu`）：

```
ids = tokenize(prompt)
prefill: forward(ids, past=0)        → 采样 next
循环:   forward([next], past=...)    → 采样 next，直到 EOS / 长度上限
```

## 5. 数值注意

- 全程 FP16 存储，但 GEMM/softmax/归一化的累加都在 FP32 进行，避免长序列
  累加误差；softmax 先减行最大值保证稳定。
- 正确性容差 atol=rtol=1e-2，逐算子与 CPU 参考（float）比对。
- 权重经 safetensors 加载；测试时先把输入"量化到 half 再转回"，让 CPU
  参考看到与 GPU 相同的输入。

## 6. 动手实验

1. 用 `make_tiny.py` 生成 tiny 模型，单步跟踪一个 block 的张量形状。
2. 关闭 causal mask，观察生成结果变化。
3. 把 GPT-2 的 GELU 换成 SiLU（仅实验），对比数值差异。
