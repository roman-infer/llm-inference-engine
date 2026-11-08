"""节点后端：复用自 _deepseek_extract/deepseek_infer（vendored 副本，勿改算法）。

本包是"节点内"算力来源（MLA / FP8 / MoE grouped GEMM / MTP / latent KV），
集群层（cluster.*）只调用、不重写。要切换为本地实时版本，把
node_backend/deepseek_infer 替换为 _deepseek_extract/deepseek_infer 即可。
"""
