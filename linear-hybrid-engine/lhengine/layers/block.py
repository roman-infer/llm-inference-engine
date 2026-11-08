# -*- coding: utf-8 -*-
"""
MixerBlock：统一封装各类型 mixer，屏蔽它们在状态结构上的差异。

不同层的状态结构不同：
    mamba1 : dict {"h": 隐状态, "conv": 卷积窗口}
    mamba2 : dict {"S": SSD 状态, "conv": 卷积窗口}
    gdn/delta/kda : Tensor（状态矩阵）
    swa    : tuple (K 窗口, V 窗口)
MixerBlock 把这些差异收敛成统一的三个方法：
    forward(x, state)  整段（prefill）
    step(x_t, state)   单步（decode）
    init_state(batch)  初始化状态
模型层只需要面对 MixerBlock，不直接接触具体 mixer。
"""
import torch
import torch.nn as nn

from ..config import (ModelConfig, MixerSpec, MIXER_SSM, MIXER_MAMBA2,
                      MIXER_DELTANET, MIXER_KDA, MIXER_SWA, MIXER_DELTA)
from .ssm import Mamba1Block
from .mamba2 import Mamba2Block
from .deltanet import GatedDeltaNetBlock
from .kda import KDABlock
from .attention import SWABlock


class MixerBlock(nn.Module):
    """单层 mixer 的统一包装。"""

    def __init__(self, spec: MixerSpec, cfg: ModelConfig):
        super().__init__()
        self.spec = spec
        self.kind = spec.kind
        d_inner = cfg.d_inner

        # 按类型构造具体实现
        if spec.kind == MIXER_SSM:
            self.impl = Mamba1Block(cfg.d_model, d_inner,
                                    d_state=cfg.state_dim_for(spec),
                                    conv_kernel=cfg.conv_kernel)
        elif spec.kind == MIXER_MAMBA2:
            self.impl = Mamba2Block(cfg.d_model, d_inner, cfg.heads,
                                    d_state=cfg.state_dim_for(spec),
                                    conv_kernel=cfg.conv_kernel)
        elif spec.kind == MIXER_DELTANET:
            self.impl = GatedDeltaNetBlock(cfg.d_model, cfg.heads,
                                           d_head=cfg.d_head, d_key=cfg.d_key,
                                           gated=True)
        elif spec.kind == MIXER_DELTA:
            self.impl = GatedDeltaNetBlock(cfg.d_model, cfg.heads,
                                           d_head=cfg.d_head, d_key=cfg.d_key,
                                           gated=False)
        elif spec.kind == MIXER_KDA:
            self.impl = KDABlock(cfg.d_model, cfg.heads, d_head=cfg.d_head,
                                 d_key=cfg.d_key)
        elif spec.kind == MIXER_SWA:
            self.impl = SWABlock(cfg.d_model, cfg.heads, d_head=cfg.d_head,
                                 n_kv_heads=cfg.n_kv_heads, window=spec.window)
        else:
            raise ValueError(f"未知 mixer 类型: {spec.kind}")

    def forward(self, x: torch.Tensor, state=None):
        """整段前向，返回 (输出, 新状态)。"""
        return self.impl(x, state=state)

    def step(self, x_t: torch.Tensor, state):
        """单步 decode。mamba1/mamba2 用 dict 状态；其余按各自接口。"""
        if self.kind in (MIXER_SSM, MIXER_MAMBA2):
            return self.impl.step(x_t, state)
        elif self.kind in (MIXER_DELTANET, MIXER_DELTA, MIXER_KDA):
            return self.impl.step(x_t, state)
        else:  # SWA
            return self.impl.step(x_t, state)

    def init_state(self, batch: int, device, dtype=torch.float32):
        """初始化该层状态。"""
        return self.impl.init_state(batch, device, dtype=dtype)
