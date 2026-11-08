"""集群级服务配置：PD 角色、EP 规模、KV 分层与 SLO。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServingConfig:
    # 节点后端
    block_size: int = 16
    dtype: str = "float16"          # float16 / float32
    device: str = "cpu"             # cpu / cuda
    seed: int = 123

    # PD 角色数量
    n_prefill: int = 1
    n_decode: int = 1

    # 专家并行 / 数据并行
    ep_size: int = 4
    dp_size: int = 1

    # KV 分层容量（块；0 = 不限）
    gpu_cap: int = 0
    cpu_cap: int = 0
    ssd_cap: int = 0
    ssd_dir: str = ""

    # SLO 与 P/D 速率匹配
    ttft_target_ms: float = 500.0
    max_inflight: int = 8           # P/D 之间在途请求上限
    prefix_reuse: bool = True
