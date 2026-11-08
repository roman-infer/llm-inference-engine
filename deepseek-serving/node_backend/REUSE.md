# 复用边界说明（node_backend）

`node_backend/deepseek_infer/` 是从 **`_deepseek_extract/deepseek_infer`** 原样拷贝
（vendored）的节点内推理库，约 1000 行，算法与数值口径保持一致，**不在本项目中改动**。

集群层直接复用的部分：

| 复用模块 | 用途 |
| --- | --- |
| `mla/*` | 每卡 MLA 注意力（prefill / decode_absorbed / chunk） |
| `fp8/*` | FP8/低比特线性层（真实 GPU 路径） |
| `moe/grouped_gemm` | EP 中每个 rank 本地专家的 grouped SwiGLU |
| `moe/moe`（路由部分） | top-k 路由与共享专家口径 |
| `engine/sampling` | 贪心 / 多项式采样 |
| `model/*`、`utils.py` | 模型结构、配置、RMSNorm、公共工具 |

被集群层**取代**（不直接使用）的部分：

- `engine/generator.py`：单序列生成循环 → 由 `cluster/engine.py` 的 PD 编排取代；
- `engine/kv_cache.py` 的管理器：latent KV **布局保留**，但稠密 buffer 由
  `cluster/kvpool/block_pool.py` 的块池 + 分层存储取代。
