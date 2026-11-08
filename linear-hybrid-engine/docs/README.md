# 数学笔记索引

本目录按"从连续动力学到现代线性注意力"的顺序，给出本项目的完整数学脉络。
建议按 L1 → L6 顺序阅读，每篇都标注了对应的代码位置。

| 笔记 | 主题 | 对应代码 |
| --- | --- | --- |
| [L1](L1_continuous_to_discrete.md) | 连续 SSM → 离散递推（ZOH） | `layers/ssm.py` |
| [L2](L2_parallel_scan.md) | 并行扫描（Hillis–Steele / 分块） | `kernels/scan.py`、`kernels/csrc/scan_kernel.cu` |
| [L3](L3_ssd_duality.md) | SSD 对偶：状态递推 ↔ 类注意力 | `layers/mamba2.py` |
| [L4](L4_delta_rule.md) | Delta rule：在线误差修正 | `layers/deltanet.py` |
| [L5](L5_wy_ut_chunkwise.md) | WY/UT：delta rule 的分块并行 | `layers/deltanet.py` |
| [L6](L6_kda_dplr.md) | KDA 与 DPLR：逐通道门控 | `layers/kda.py` |

## 一条主线

```
连续 SSM ──离散化──> 时变递推 ──关联扫描──> 并行
                                          │
              ┌───────────────────────────┴───────────────────────────┐
        Mamba2 SSD（矩阵对偶）                          DeltaNet（误差修正）
              │                                              │
        块内并行 + 固定状态                            WY/UT 分块 + 固定状态
              └───────────────────┬────────────────────────────┘
                                  ▼
                        KDA：逐通道门控（DPLR）
                        固定大小 state，无 KV 增长
```

## 核心心智模型

> Transformer 把历史存成**随长度增长的 KV**（$O(N)$ 显存）；
> 线性/混合架构把历史压缩进**固定大小的状态矩阵**（$O(1)$ 显存），
> 而 prefill 仍可通过分块并行达到 $O(N)$ 的计算复杂度。
