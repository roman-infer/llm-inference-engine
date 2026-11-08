"""请求与生命周期。"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Request:
    rid: str
    prompt: torch.Tensor            # 1D token ids
    max_new: int
    arrival: int = 0
    status: str = "queued"          # queued / prefilling / decoding / done
    output: list = field(default_factory=list)
    prefix_blocks: int = 0
