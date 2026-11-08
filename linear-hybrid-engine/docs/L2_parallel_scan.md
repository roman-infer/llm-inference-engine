# L2 · 并行扫描（Parallel Scan）

> 对应代码：`lhengine/kernels/scan.py`（PyTorch）与 `lhengine/kernels/csrc/scan_kernel.cu`（CUDA）。
> 目标：把一个看似"只能逐步循环"的递推，用对数级深度的并行算法算出来。

## 1. 问题

给定时变递推（每个通道独立）：

$$
h_t = a_t h_{t-1} + b_t,\qquad h_{-1}=0
$$

朴素循环需要 $L$ 步串行。但它本质是对"变换"的**关联运算**，可以并行。

## 2. 把递推看成变换的复合

记单步变换为二元组 $T_t=(a_t,b_t)$，含义是 $h\mapsto a_t h+b_t$。
两个变换的复合（先经过 $T_f$、再经过 $T_n$）：

$$
T_n\circ T_f
= (a_n,b_n)\circ(a_f,b_f)
= (a_n a_f,\; a_n b_f+b_n)
$$

**关键**：该算子满足结合律：

$$
(T_1\circ T_2)\circ T_3 = T_1\circ(T_2\circ T_3)
$$

因此"前 $t$ 步的总变换"是一个前缀和（prefix sum）问题，可以并行求。

## 3. Hillis–Steele 扫描（块内）

对长度 $L$，做 $\lceil\log_2 L\rceil$ 轮，第 $k$ 轮步长 $2^k$：

$$
T_i \leftarrow T_i\circ T_{i-2^k}\quad(i\ge 2^k)
$$

每轮所有位置并行，整体深度 $O(\log L)$、工作量 $O(L\log L)$。

### 小例子（$L=4$）

- 初始：$T_0,T_1,T_2,T_3$
- 步长1：$T_1=T_1\circ T_0$；$T_2=T_2\circ T_1$；$T_3=T_3\circ T_2$
- 步长2：$T_2=T_2\circ T_0$；$T_3=T_3\circ T_1$
- 结束：$T_i$ 即前 $i$ 步总变换，其 $b$ 分量就是 $h_i$。

## 4. 分块扫描（Blelloch 风格，长序列）

当 $L$ 超过一个 block 的容量，采用三段式：

1. **块内扫描**：每个 block 对自己的 chunk 做 Hillis–Steele，并算出该 chunk 的"总 carry"
   $C_q=(A_q,B_q)$；
2. **carry 扫描**：对所有 chunk 的 carry 做（顺序/并行）扫描，得到每个 chunk 左侧的前缀 carry；
3. **应用 carry**：把前缀 carry 作用到块内结果：

$$
h = A_{\text{pre}}\,h_{\text{partial}}+B_{\text{pre}}
$$

这样块内完全并行、块间只需传播少量 carry，是 GPU 上的标准做法。

## 5. 两种模式：recurrent vs chunkwise

- **Recurrent（循环）**：$O(L)$ 步，每步 $O(1)$；decode 时天然适配（每次只来一个 token）；
- **Chunkwise（分块）**：块内并行，prefill 时一次性处理长序列，吞吐高。

二者在数学上**严格等价**（测试 `test_*_recurrent_vs_chunkwise` 验证）。

## 小结

| 概念 | 内容 |
| --- | --- |
| 关联算子 | $(a_n,b_n)\circ(a_f,b_f)=(a_na_f,a_nb_f+b_n)$ |
| 块内 | Hillis–Steele，深度 $O(\log L)$ |
| 块间 | carry 扫描 + 应用 |
| 意义 | 递推从"串行 $L$ 步"变成"并行 $\log L$ 深度" |
