# -*- coding: utf-8 -*-
"""layers 子包：各类型 mixer 层与统一包装。"""
from .common import RMSNorm, CausalConv1d, silu_gate
from .ssm import Mamba1Block, selective_ssm_core
from .mamba2 import Mamba2Block, ssd_recurrent, ssd_chunkwise
from .deltanet import GatedDeltaNetBlock, delta_recurrent, delta_chunkwise
from .kda import KDABlock, kda_recurrent, kda_block_loop
from .attention import SWABlock, sliding_window_mask
from .block import MixerBlock

__all__ = [
    "RMSNorm", "CausalConv1d", "silu_gate",
    "Mamba1Block", "selective_ssm_core",
    "Mamba2Block", "ssd_recurrent", "ssd_chunkwise",
    "GatedDeltaNetBlock", "delta_recurrent", "delta_chunkwise",
    "KDABlock", "kda_recurrent", "kda_block_loop",
    "SWABlock", "sliding_window_mask",
    "MixerBlock",
]
