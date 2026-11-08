# L6 · KDA 与 DPLR：Channel-wise 门控

> 对应代码：`lhengine/layers/kda.py`；参考 Kimi Linear 技术报告（arXiv:2510.26692）。
> 目标：讲清 Kimi Delta Attention（KDA）相比 Gated DeltaNet 的升级——把门控从"标量"
> 做到"逐通道"，以及它与 DPLR（对角加低秩）结构的关系。

## 1. Gated DeltaNet 的局限

Gated DeltaNet 的遗忘门是**每头一个标量**：

$$
S_t = \alpha_t S_{t-1} + \beta_t r_t k_t^{\top},\qquad \alpha_t\in\mathbb{R}
$$

一个标量只能对整个状态做"统一衰减"，无法对不同特征通道差异化地保留/遗忘。

## 2. KDA：Channel-wise 门控

Kimi Linear 把衰减门做成**逐通道向量**（对 value/head 维 $P$ 分别门控）：

$$
\boxed{
\begin{aligned}
\boldsymbol{\alpha}_t &= \text{sigmoid}(W_\alpha x_t)\quad\in\mathbb{R}^{P}
&\text{（每通道一个门）}\\
\bar v_t &= (\boldsymbol{\alpha}_t\odot S_{t-1}k_t)\\
r_t &= v_t-\bar v_t\\
S_t &= \boldsymbol{\alpha}_t\odot S_{t-1} + \beta_t r_t k_t^{\top}
\end{aligned}}
$$

- $\boldsymbol{\alpha}_t$ 是通道维向量，相当于一个**对角衰减矩阵**
  $\text{diag}(\boldsymbol{\alpha}_t)$；
- 不同通道可以有不同的"记忆时长"，表达能力显著增强；
- 这是 Kimi Linear（2025.10 开源）的核心算子，Kimi K3 全面采用。

## 3. 与 DPLR 的关系

经典 SSM（S4）的连续状态矩阵常写成 **DPLR（Diagonal Plus Low-Rank）**：

$$
A = \Lambda - B C^{\top}
$$

- $\Lambda$ 是对角部分（对应逐通道的衰减/门控）；
- $BC^{\top}$ 是低秩修正（对应输入驱动的状态更新）。

KDA 可以看作这一结构在 delta-rule 框架下的离散对应：

- 对角项 $\text{diag}(\boldsymbol{\alpha}_t)$：逐通道独立衰减；
- 低秩项 $\beta_t r_t k_t^{\top}$：rank-1 的数据相关写入。

这正是"门控线性注意力 + 结构化状态"在现代架构里的统一形态。

## 4. 标量门是 KDA 的特例

当所有通道门相等（$\boldsymbol{\alpha}_t=\alpha_t\mathbf{1}$）时，KDA 退化为
Gated DeltaNet。测试 `test_scalar_gate_reduces_to_gdn` 验证这一退化关系，
保证两种实现自洽。

## 5. 为什么前沿且重要

- **显存**：依旧是固定大小状态（$P\times K$），decode 无 KV 增长；
- **表达力**：逐通道门控逼近 DPLR 的容量，又保持线性可并行；
- **工程**：Kimi 已开源 kernel + vLLM 实现，是"后 Transformer 落地"的代表；
- 长上下文场景下，同时拿到 Transformer 级质量与线性架构的 $O(1)$ 状态。

## 小结

| 维度 | Gated DeltaNet | KDA |
| --- | --- | --- |
| 门控 | 标量 $\alpha_t$ | 向量 $\boldsymbol{\alpha}_t$（逐通道） |
| 对应结构 | 单衰减 | 对角（DPLR 的 $\Lambda$） |
| 表达力 | 统一遗忘 | 通道级差异化记忆 |
| 关系 | KDA 的特例 | 推广 |
