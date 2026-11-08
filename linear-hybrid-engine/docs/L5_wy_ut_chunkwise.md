# L5 · WY / UT 表示：把 Delta Rule 分块并行

> 对应代码：`lhengine/layers/deltanet.py`（`delta_chunkwise`、`_ut_transform`）。
> 目标：讲清 Gated DeltaNet 如何把"逐 token 的 delta 递推"变成"块内一次矩阵求解"。

## 1. 动机

Delta rule 的递推是顺序的（$S_t$ 依赖 $S_{t-1}$）。prefill 长序列时，需要像
SSD 一样分块并行。关键工具是 **WY 表示**（得名于 Woodbury 与一个紧凑矩阵形式，
在 fast weight programmer 文献中广泛使用）。

## 2. 块内递推的紧凑形式

考虑一个 chunk（长度 $c$），把每一步的更新写成：

$$
S_t = S_{t-1}(I-\beta_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}
$$

（这里用右乘形式，与论文记号一致；代码里用 einsum 实现，数学等价。）

累积一个 chunk 后，状态可以表示为：

$$
S_{\text{out}} = S_{\text{in}}\,W + U
$$

其中 $W$（对 key 空间的"擦除矩阵"）和 $U$（新写入的 value 矩阵）只依赖于块内的
$K,V,\beta$，**与 $S_{\text{in}}$ 无关**，因此可以预先并行算好。

## 3. WY 表示（ungated）

定义累积矩阵：

$$
W = I - K^{\top}T K,\qquad U = T V
$$

其中 $T$ 是一个**下三角**矩阵，由 $\beta$ 递推得到：

$$
T = \text{strictLower}(T\cdot\text{diag}(\beta)\,K^{\top}K)+\text{diag}(\beta)
$$

直观上：

- $T$ 是一个"块内 delta 更新算子"，下三角体现因果（位置 $i$ 只受 $j\le i$ 影响）；
- $W$ 是整个 chunk 对 key 空间的净擦除；
- $U$ 是 chunk 净写入的 value。

二者都可以通过**解一个下三角线性方程组**得到：

$$
T:\quad \big(I+\text{strictLower}(\text{diag}(\beta)K^{\top}K)\big)T^{\top}
       =\text{diag}(\beta)
$$

这是 $O(c^2 K)$ 的三角求解，可用 `torch.linalg.solve`/`solve_triangular` 高效完成。

## 4. Gated（UT）表示：必须把 $W$ 与 $U$ 分开

加入遗忘门 $\alpha$（衰减 $\Gamma$）后，论文的 gated chunkwise 有一个**关键细节**：

- 块内输出项 $\tilde U$ 使用**含衰减**的算子 $T_g$（其中 key 乘积被 $\Gamma$ 加权）；
- 但 $W$ 必须使用**不含衰减**的 $T_{\text{ung}}$，外层再统一乘衰减 $\gamma$。

若 $W$ 也用 $T_g$，衰减会被重复计算（在 $t=2$ 会多出一个 $\alpha_2$ 因子）。

代码中 `_ut_transform` 严格区分：

$$
\begin{aligned}
W &= \text{solve}\big(I+\text{strictLower}(\text{diag}\beta\,KK),\ \text{diag}\beta\big)\cdot K
&&\text{（ungated）}\\
U &= \text{solve}\big(I+\text{strictLower}(\text{diag}\beta\,(\Gamma\odot KK)),\ \text{diag}\beta\big)\cdot V
&&\text{（gated）}
\end{aligned}
$$

## 5. 块内输出

gated 版本的块内输出，其掩码是**衰减矩阵** $\Gamma[r,i]=\gamma_r/\gamma_i$：

$$
y_r = \gamma_r S_{\text{in}}q_r + \sum_{i\le r}\frac{\gamma_r}{\gamma_i}R_i(k_i q_r)
$$

ungated 时 $\gamma=1$，自动退化为普通因果掩码。

## 6. 为什么这样就并行

- 块内：$W,U$ 通过三角求解一次性算出（GPU 并行的矩阵运算）；
- 块间：$S_{\text{out}}=S_{\text{in}}W+U$ 是逐块传播，块数少；
- 与逐步递推严格等价（测试 `test_recurrent_vs_chunkwise_gated` 验证）。

## 小结

| 符号 | 含义 |
| --- | --- |
| $T$ | 块内因果更新算子（下三角） |
| $W=I-K^\top T K$ | 块内净擦除（key 空间） |
| $U=TV$ | 块内净写入（value） |
| gated 关键 | $W$ 用 ungated $T$，$U$ 用 gated $T$ |
| 块内输出掩码 | 衰减矩阵 $\Gamma$ |
