"""DeepSeek MTP head：随模型训练的多 token 预测模块。

结构（DeepSeek-V3 MTP）：
    h' = RMSNorm(W_h h_prev + W_e emb(next_token))
    h'' = 一个独立 Transformer block（MoE）
    logits = lm_head(h'')，lm_head 与主干共享
推理时自回归复用该模块产出草稿 token。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.config import ModelConfig
from ..model.model import Block
from ..model.norms import RMSNorm


def run_block(block: Block, x: torch.Tensor, moe_mode: str = "grouped") -> torch.Tensor:
    """无 cache 的单层 block 前向。x: (1,S,d)。

    moe_mode: 推理用 "grouped"（argmax dispatch，快但不可微）；
              训练用 "naive"（全专家连续加权，可微）。
    """
    an = block.attn_norm(x)
    attn_out, _, _ = block.attn.forward_prefill(an)
    x = x + attn_out
    return x + block.apply_ffn(block.ffn_norm(x), mode=moe_mode)


class MTPHead(nn.Module):
    def __init__(self, cfg: ModelConfig, embed_tokens: nn.Embedding, head_weight: torch.Tensor,
                 device="cuda", dtype=torch.float16):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = embed_tokens
        self.register_buffer("head_weight", head_weight, persistent=False)
        self.w_h = nn.Linear(cfg.d_model, cfg.d_model, bias=False, device=device, dtype=dtype)
        self.w_e = nn.Linear(cfg.d_model, cfg.d_model, bias=False, device=device, dtype=dtype)
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps, device, dtype)
        self.block = Block(cfg, is_dense=False, device=device, dtype=dtype)
        for p in (self.w_h, self.w_e):
            nn.init.normal_(p.weight, std=cfg.initializer_range)

    def forward(self, h_prev: torch.Tensor, next_token: torch.Tensor,
                moe_mode: str = "grouped"
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """h_prev: (1,S,d) 或 (1,d)；next_token: 标量或 (S,)。"""
        if h_prev.dim() == 2:
            h_prev = h_prev.unsqueeze(0)
        e = self.embed_tokens(next_token.view(-1)).unsqueeze(0)   # (1,S,d)
        x = self.norm(self.w_h(h_prev) + self.w_e(e))
        h = run_block(self.block, x, moe_mode=moe_mode)
        logits = F.linear(h, self.head_weight)
        return h, logits
