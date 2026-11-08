# L3 · SSD：状态空间的"对偶"

> 对应代码：`lhengine/layers/mamba2.py`（`ssd_recurrent`、`ssd_chunkwise`）。
> 目标：讲清 Mamba2 为什么同时是"状态递推"和"类注意力矩阵"，以及它如何二者兼得。

## 1. 多头 SSD 递推

Mamba2 把 Mamba1 的"每维一个 $\Delta$"简化为"每头一个标量 $\Delta$"，并让 $B,C$ 也多头。
对单个头（state 维 $N$，head 维 $P$）：

$$
\begin{aligned}
S_t &= \overline{A}_t\,S_{t-1} + B_t\,x_t^{\top}
      &&\text{（$S$ 是 $N\times P$ 矩阵）}\\
y_t &= C_t\,S_t
\end{aligned}
$$

其中 $\overline{A}_t=\text{diag}(e^{\Delta_t A})$。

## 2. 展开成"注意力"形式

把递推展开（令 $S_{-1}=0$）：

$$
y_t = \sum_{j\le t} C_t\,\underbrace{\Big(\textstyle\prod_{k=j+1}^{t}\overline{A}_k\Big)}_{M[t,j]}\,B_j\,x_j
$$

定义衰减矩阵：

$$
M[t,j] = \prod_{k=j+1}^{t}\overline{A}_k
        = \exp\big(\text{cum}_t-\text{cum}_j\big)
$$

则：

$$
y_t = \sum_{j\le t}\big(C_t M[t,j] B_j\big)x_j
$$

这与注意力的 $\sum_{j}\text{softmax}(q_t k_j)v_j$ 形式一模一样，只是：

- 没有 softmax（线性）；
- $M$ 是数据相关的**因果衰减**矩阵。

## 3. 两种计算视角（对偶）

| 视角 | 计算 | 适用 |
| --- | --- | --- |
| **递推（recurrent）** | 维护固定大小 $S$，逐步更新 | decode，$O(1)$ 状态 |
| **分块（chunkwise）** | 块内用矩阵 $M$ 像注意力一样并行算 | prefill，高并行 |

二者数学等价，这就是"Structured State Space **Duality**"的含义：
**同一套计算，既可按状态递推执行，也可按注意力形式并行。**

## 4. 分块公式

对一个 chunk（块内位置 $r$，块号 $q$）：

$$
\begin{aligned}
\text{cum} &= \text{cumsum}(\log\overline{A})
&\text{块内前缀}\\
M[i,j] &= \exp(\text{cum}_i-\text{cum}_j)
&\text{块内衰减}\\
Y_{\text{intra}} &= (M\odot C B^{\top})X
&\text{块内输出}\\
Y_{\text{inter}} &= C\cdot(\exp(\text{cum})\,S)
&\text{跨块（历史状态）输出}\\
S' &= \exp(\text{cum}_{-1})S + \big(\exp(\text{cum}_{-1}-\text{cum})\odot B\big)X
&\text{块末状态}
\end{aligned}
$$

## 5. 数值稳定性要点（对应已修复的 bug）

1. **未来位置必须先填 $-\infty$ 再取 exp**：
   对 $j>i$，$\text{cum}_i-\text{cum}_j$ 是大正数，直接 exp 会得到 $\inf$，
   再乘因果掩码 $0$ 变成 $\inf\times0=\text{NaN}$。正确做法是先掩码成 $-\infty$，exp 得 $0$。
2. **float32 累加**，避免长序列衰减误差累积。

## 小结

- SSD = 线性注意力 + 因果衰减；
- 递推形态给 decode 带来 $O(1)$ 固定状态；
- 分块形态给 prefill 带来 $O(L)$ 并行；
- 这是 Mamba2 相比 Mamba1 "更快又更省"的根本原因。
