"""DeepSeek 风格模型：MLA Block（首层 dense FFN，其余 MoE）+ MTP 挂载点。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mla import MLA, MLAConfig
from ..moe import MoE, MoEConfig
from .config import ModelConfig
from .norms import RMSNorm, swiglu


class DenseFFN(nn.Module):
    def __init__(self, d: int, inter: int, device="cuda", dtype=torch.float16, std=0.08):
        super().__init__()
        self.wg = nn.Parameter(torch.empty(inter, d, device=device, dtype=dtype))
        self.wu = nn.Parameter(torch.empty(inter, d, device=device, dtype=dtype))
        self.wd = nn.Parameter(torch.empty(d, inter, device=device, dtype=dtype))
        for p in self.parameters():
            nn.init.normal_(p, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x, self.wg, self.wu, self.wd)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, is_dense: bool, device="cuda", dtype=torch.float16):
        super().__init__()
        mla_cfg = MLAConfig(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_head_rope,
                            cfg.d_latent, cfg.d_latent_q)
        self.attn = MLA(mla_cfg, cfg.max_seq_len, device, dtype)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps, device, dtype)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps, device, dtype)
        if is_dense:
            self.ffn = DenseFFN(cfg.d_model, cfg.dense_inter, device, dtype,
                                std=cfg.initializer_range)
        else:
            moe_cfg = MoEConfig(cfg.d_model, cfg.moe_inter, cfg.n_routed,
                                cfg.n_active, cfg.n_shared)
            self.ffn = MoE(moe_cfg, device, dtype)
        self.is_dense = is_dense

    def apply_ffn(self, x: torch.Tensor, mode: str = "grouped") -> torch.Tensor:
        """自动处理 MoE reshape 的 FFN 前向。x: (..., d)。"""
        if self.is_dense:
            return self.ffn(x)
        orig_shape = x.shape
        return self.ffn(x.reshape(-1, orig_shape[-1]), mode=mode).reshape(orig_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # 由 Model 统一驱动（含 cache）


class DeepSeekModel(nn.Module):
    def __init__(self, cfg: ModelConfig, device="cuda", dtype=torch.float16):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model, device=device, dtype=dtype)
        nn.init.normal_(self.embed_tokens.weight, std=cfg.initializer_range)
        self.layers = nn.ModuleList([
            Block(cfg, is_dense=(i == 0), device=device, dtype=dtype)
            for i in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps, device, dtype)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False,
                                     device=device, dtype=dtype)
            nn.init.normal_(self.lm_head.weight, std=cfg.initializer_range)
        self.cache = None  # 由 engine 绑定

    def _head(self) -> torch.Tensor:
        return self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight

    # ---------- PREFILL ----------
    def forward_prefill(self, tokens: torch.Tensor, positions: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: (B,S) 或 (S,)。返回 logits, last_hidden(final norm 后)。"""
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        b, s = tokens.shape
        assert b == 1, "prefill 仅支持 batch=1"
        h = self.embed_tokens(tokens)
        if positions is None:
            positions = torch.arange(s, device=tokens.device)
        for li, layer in enumerate(self.layers):
            an = layer.attn_norm(h)
            attn_out, ckv, krope = layer.attn.forward_prefill(an, positions)
            if self.cache is not None:
                self.cache.set_layer(li, ckv[0], krope[0])
            h = h + attn_out
            h = h + layer.apply_ffn(layer.ffn_norm(h))
        h = self.final_norm(h)
        logits = F.linear(h, self._head())
        return logits, h

    # ---------- 继续前向（在已有 cache 上追加 chunk，用于投机验证）----------
    def forward_continue(self, tokens: torch.Tensor, start_position: int
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: (S,)，全局位置 start_position..+S-1；cache 已有其之前历史。"""
        s = tokens.shape[0]
        positions = torch.arange(start_position, start_position + s, device=tokens.device)
        h = self.embed_tokens(tokens.view(1, s))
        for li, layer in enumerate(self.layers):
            prefix_ckv = self.cache.ckv[li, :start_position]
            prefix_krope = self.cache.krope[li, :start_position]
            an = layer.attn_norm(h)
            attn_out, cckv, ckrope = layer.attn.forward_chunk(
                an, prefix_ckv, prefix_krope, positions)
            self.cache.ckv[li, start_position:start_position + s].copy_(cckv)
            self.cache.krope[li, start_position:start_position + s].copy_(ckrope)
            h = h + attn_out
            h = h + layer.apply_ffn(layer.ffn_norm(h))
        h = self.final_norm(h)
        return F.linear(h, self._head()), h

    # ---------- DECODE（单序列单 token）----------
    def forward_decode(self, token: torch.Tensor, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """token: 标量张量。返回 logits(1,vocab), hidden(1,d)。"""
        h = self.embed_tokens(token.view(1, 1))  # (1,1,d)
        for li, layer in enumerate(self.layers):
            an = layer.attn_norm(h)[0, 0]       # (d,)
            ckv_hist = self.cache.ckv[li, :position]
            krope_hist = self.cache.krope[li, :position]
            attn_out = layer.attn.decode_absorbed(an, ckv_hist, krope_hist, position)
            ckv_t, krope_t = layer.attn._cache_token(an, position)
            self.cache.set_token(li, position, ckv_t, krope_t)
            h = h + attn_out.view(1, 1, -1)
            h = h + layer.apply_ffn(layer.ffn_norm(h))
        h = self.final_norm(h)
        logits = F.linear(h, self._head())
        return logits, h
