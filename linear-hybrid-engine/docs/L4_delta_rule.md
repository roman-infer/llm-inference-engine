# L4 · Delta Rule（Delta 规则）

> 对应代码：`lhengine/layers/deltanet.py`（`delta_recurrent`）。
> 目标：讲清 DeltaNet 如何把"在线学习的误差修正"写进固定大小状态。

## 1. 联想记忆与线性注意力

线性注意力把序列存进一个关联矩阵（状态）：

$$
S_t = S_{t-1} + v_t k_t^{\top},\qquad y_t = S_t q_t
$$

这是"只做加法"的记忆：同一个 key 反复出现时，旧 value 不会被覆盖，只会叠加，
导致记忆"污染"。

## 2. Delta rule：先擦后写

DeltaNet 借鉴在线梯度下降（Widrow–Hoff / delta rule），在写入前先"擦除"该 key
对应的旧值：

$$
\begin{aligned}
\beta_t &= \text{sigmoid}(W_\beta x_t)
&\text{（写入强度）}\\
v^{\text{old}}_t &= S_{t-1} k_t
&\text{（当前 key 下记忆里的旧 value）}\\
r_t &= v_t - v^{\text{old}}_t
&\text{（预测误差 / 要更新的量）}\\
S_t &= S_{t-1} + \beta_t\,r_t k_t^{\top}
\end{aligned}
$$

把更新展开：

$$
S_t = (I-\beta_t k_t k_t^{\top})S_{t-1} + \beta_t v_t k_t^{\top}
$$

这正是对损失 $\tfrac12\|v-Sk\|^2$ 做一步梯度下降的结果，所以叫 delta rule。

## 3. 为什么必须 L2 归一化 key/query

更新矩阵 $I-\beta kk^{\top}$ 的稳定性取决于 $\|k\|$：

- 若 $k$ 未归一化，$\|k\|^2$ 可能很大，使 $I-\beta kk^{\top}$ 的特征值
  $1-\beta\|k\|^2$ 绝对值大于 1，状态**指数爆炸**；
- 对 $k,q$ 做 L2 归一化（$\|k\|=\|q\|=1$），则 $\beta\|k\|^2\in[0,1]$，
  特征值稳定落在 $[0,1]$，递推收敛。

论文把 L2 normalization 列为**必要**步骤，代码在 `_split` 中对 $K,Q$ 做了归一化。

## 4. Gated DeltaNet：加一个遗忘门

在 delta rule 基础上引入标量衰减门 $\alpha_t=\text{sigmoid}(W_\alpha x_t)$：

$$
\boxed{
\begin{aligned}
\bar v_t &= \alpha_t\,S_{t-1}k_t \\
r_t &= v_t-\bar v_t \\
S_t &= \alpha_t S_{t-1} + \beta_t r_t k_t^{\top}
\end{aligned}}
$$

- $\alpha_t$ 控制"记住多少过去"（类似 RNN 的遗忘门）；
- $\alpha_t=1$ 时退化为普通 DeltaNet；
- 它把"线性注意力的衰减"和"delta rule 的误差修正"结合起来。

## 5. 与线性注意力的关系

| 模型 | 状态更新 |
| --- | --- |
| 线性注意力 | $S_t=S_{t-1}+v_t k_t^\top$ |
| DeltaNet | $S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top$ |
| Gated DeltaNet | $S_t=\alpha_t S_{t-1}+\beta_t(v_t-\alpha_t S_{t-1}k_t)k_t^\top$ |

Delta rule 让状态成为一个**可自我纠错的联想记忆**，这是它比普通线性注意力更强的关键。

## 小结

- delta rule = 对联想记忆做在线梯度下降；
- $r_t=v_t-Sk_t$ 是"误差"；
- L2 归一化保证递推稳定；
- 门控版本再加遗忘因子 $\alpha$。
