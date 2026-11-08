# L1 · 从连续 SSM 到离散递推

> 对应代码：`lhengine/layers/ssm.py`（`selective_ssm_core`）。
> 目标：讲清"为什么一个连续的微分方程，在计算机里变成了 `h_t = dA·h_{t-1}+dB·x`"。

## 1. 连续状态空间模型

连续 SSM 用一个线性时不变（LTI）的微分方程描述隐状态 $h(t)\in\mathbb{R}^{N}$：

$$
\begin{aligned}
h'(t) &= A\,h(t) + B\,x(t) \\
y(t) &= C\,h(t) + D\,x(t)
\end{aligned}
$$

- $A\in\mathbb{R}^{N\times N}$（本项目取对角 $A$，即每维独立）；
- $B\in\mathbb{R}^{N\times 1}$、$C\in\mathbb{R}^{1\times N}$；
- $D$ 是 skip（残差）项。

## 2. 解析解

在区间 $[0,t]$ 上，利用矩阵指数 $e^{At}$，解为：

$$
h(t) = e^{At}h(0) + \int_{0}^{t} e^{A(t-s)}B\,x(s)\,ds
$$

## 3. 离散化（Zero-Order Hold, ZOH）

假设输入在一个时间步 $\Delta$ 内保持不变：$x(s)=x_t$（$s\in[t\Delta,(t+1)\Delta)$）。
对解析解在一个步长上积分，得到**精确**的离散递推：

$$
\begin{aligned}
\overline{A} &= e^{\Delta A} \\
\overline{B} &= \left(\int_{0}^{\Delta} e^{A\tau}\,d\tau\right) B
\end{aligned}
$$

于是：

$$
h_t = \overline{A}\,h_{t-1} + \overline{B}\,x_t,\qquad
y_t = C h_t + D x_t
$$

### 教学简化

代码里用 $\overline{B}\approx(\Delta A)^{-1}(e^{\Delta A}-I)B$，并在一阶近似下
简化为 $\overline{B}\approx\Delta B$（当 $\Delta A$ 较小）。这是 Mamba 实现里
常用的简化，数值上足够接近。

## 4. 为什么 $A$ 取负数

为了让系统"稳定且有衰减记忆"，需要 $e^{\Delta A}$ 的特征值落在 $(0,1)$，即 $A$ 的
实部为负。Mamba 用参数化：

$$
A = -\exp(A_{\log})
$$

这样 $A$ 恒负，$\overline{A}=e^{\Delta A}\in(0,1)$，越久远的信息衰减越多。

## 5. 选择性（Mamba1）：参数随输入变化

经典 SSM 里 $\Delta,B,C$ 是固定的；Mamba1 让它们**由当前输入动态产生**：

$$
\Delta_t = \text{softplus}(W_\Delta x_t),\quad B_t = W_B x_t,\quad C_t = W_C x_t
$$

于是离散化也变成"时变"的：

$$
\boxed{\,h_t = e^{\Delta_t A}\,h_{t-1} + (\Delta_t B_t)\,x_t\,}
$$

这个"时变一阶递推"正是 L2 要并行扫描的对象。

## 小结

| 连续 | 离散（ZOH） |
| --- | --- |
| $h'=Ah+Bx$ | $h_t=\overline A h_{t-1}+\overline B x_t$ |
| $\overline A=e^{\Delta A}$ | 衰减系数 |
| $\overline B=(\int_0^\Delta e^{A\tau}d\tau)B\approx\Delta B$ | 输入系数 |
