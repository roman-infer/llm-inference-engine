"""Multi-head Latent Attention (DeepSeek MLA)。

三条计算路径：
- forward_prefill: 完整 MLA，输出结果与应缓存的 (c_kv, k_rope)；KV 只缓存压缩潜变量
  c_kv (d_c) 与解耦 RoPE key (n_h*d_hR)，而非完整 k/v。
- decode_naive: 从缓存的 c_kv 重建全部 k_nope/v，再做单 query 注意力（基线）。
- decode_absorbed: 权重吸收（weight absorption）——利用矩阵乘结合律：
      q·k_t = q·W_UK c_t = (W_UK^T q)·c_t
      o = Σ w_t W_UV c_t = W_UV(Σ w_t c_t)
  直接在 d_c 潜变量空间做注意力，最后一次上投影，避免物化历史 k_nope/v。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .rope import apply_rope, precompute_rope


@dataclass
class MLAConfig:
    d: int          # hidden size
    n_h: int        # 注意力头数
    d_h: int        # 每头 nope 维度
    d_hR: int       # 每头 rope 维度
    d_c: int        # KV 压缩潜变量维度
    d_cq: int       # q 压缩潜变量维度（通常 = d_c）

    @property
    def head_total(self) -> int:
        return self.d_h + self.d_hR

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> dict:
        mla = (self.d_c + self.n_h * self.d_hR) * dtype_bytes
        mha = 2 * self.n_h * self.head_total * dtype_bytes
        return {"mla_bytes": mla, "mha_bytes": mha, "compression": mha / mla}


class MLA(torch.nn.Module):
    def __init__(self, cfg: MLAConfig, max_seq_len: int = 4096, device="cuda",
                 dtype=torch.float16):
        super().__init__()
        self.cfg = cfg
        c = cfg
        scale = 0.08
        self.w_dkv = torch.nn.Parameter(torch.empty(c.d_c, c.d, device=device, dtype=dtype))
        self.w_uk = torch.nn.Parameter(torch.empty(c.n_h * c.d_h, c.d_c, device=device, dtype=dtype))
        self.w_uv = torch.nn.Parameter(torch.empty(c.n_h * c.d_h, c.d_c, device=device, dtype=dtype))
        self.w_dq = torch.nn.Parameter(torch.empty(c.d_cq, c.d, device=device, dtype=dtype))
        self.w_uq = torch.nn.Parameter(
            torch.empty(c.n_h * c.head_total, c.d_cq, device=device, dtype=dtype))
        self.w_kr = torch.nn.Parameter(torch.empty(c.n_h * c.d_hR, c.d, device=device, dtype=dtype))
        self.w_o = torch.nn.Parameter(torch.empty(c.d, c.n_h * c.d_h, device=device, dtype=dtype))
        for p in self.parameters():
            torch.nn.init.normal_(p, std=scale)
        cos, sin = precompute_rope(c.d_hR, max_seq_len, device=device, dtype=dtype)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ---------- 公共：由 h 计算 q/k/v 组件 ----------
    def _project(self, h: torch.Tensor):
        c = self.cfg
        ckv = F.linear(h, self.w_dkv)                                   # (...,d_c)
        knope = F.linear(ckv, self.w_uk).view(*h.shape[:-1], c.n_h, c.d_h)
        v = F.linear(ckv, self.w_uv).view(*h.shape[:-1], c.n_h, c.d_h)
        cq = F.linear(h, self.w_dq)
        qall = F.linear(cq, self.w_uq).view(*h.shape[:-1], c.n_h, c.head_total)
        qnope = qall[..., : c.d_h]
        qrope_pre = qall[..., c.d_h:]
        krope_pre = F.linear(h, self.w_kr).view(*h.shape[:-1], c.n_h, c.d_hR)
        return ckv, knope, v, qnope, qrope_pre, krope_pre

    def _q_single(self, h: torch.Tensor, position: int):
        """单 token 的 q（含 rope）。h: (d,)。返回 qfull (n_h,head_total), qnope, qrope。"""
        c = self.cfg
        cq = F.linear(h, self.w_dq)
        qall = F.linear(cq, self.w_uq).view(c.n_h, c.head_total)
        qnope = qall[:, : c.d_h]
        qrope = apply_rope(qall[:, c.d_h:].unsqueeze(0), self.rope_cos, self.rope_sin,
                           positions=torch.tensor([position], device=h.device)).squeeze(0)
        qfull = torch.cat((qnope, qrope), dim=-1)
        return qfull, qnope, qrope

    def _cache_token(self, h_t: torch.Tensor, position: int):
        """计算当前 token 的 latent cache 项。返回 (ckv_t (d_c,), krope_t (n_h,d_hR))。"""
        c = self.cfg
        ckv_t = F.linear(h_t, self.w_dkv)
        krope_pre = F.linear(h_t, self.w_kr).view(c.n_h, c.d_hR)
        krope_t = apply_rope(krope_pre.unsqueeze(0), self.rope_cos, self.rope_sin,
                              positions=torch.tensor([position], device=h_t.device)).squeeze(0)
        return ckv_t, krope_t

    def _rope_qk(self, qnope, qrope_pre, krope_pre, positions):
        """对 q/k 施加 RoPE 并拼接。返回 (qfull, krope)。"""
        qrope = apply_rope(qrope_pre, self.rope_cos, self.rope_sin, positions)
        krope = apply_rope(krope_pre, self.rope_cos, self.rope_sin, positions)
        return torch.cat((qnope, qrope), dim=-1), krope

    # ---------- PREFILL ----------
    def forward_prefill(self, h: torch.Tensor, positions: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h: (B,S,d)。返回 out(B,S,d), ckv(B,S,d_c), krope(B,S,n_h,d_hR)。"""
        c = self.cfg
        ckv, knope, v, qnope, qrope_pre, krope_pre = self._project(h)
        qfull, krope = self._rope_qk(qnope, qrope_pre, krope_pre, positions)
        kfull = torch.cat((knope, krope), dim=-1)
        scale = 1.0 / c.head_total ** 0.5
        # 显式注意力（数值可核对；小配置下与 sdpa 一致）
        from .attention import attention_naive
        o = attention_naive(qfull, kfull, v, causal=True, scale=scale)
        o = o.reshape(*h.shape[:-1], c.n_h * c.d_h)
        out = F.linear(o, self.w_o)
        return out, ckv, krope

    # ---------- CHUNK（带历史 prefix，用于 target 批量验证草稿）----------
    def forward_chunk(self, h: torch.Tensor, prefix_ckv: torch.Tensor,
                      prefix_krope: torch.Tensor, positions: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h: (1,S,d) 当前窗口；prefix_*: (T,..) 历史 latent（T 可为 0）。

        当前窗口的 q 可见全部历史与窗口内 causal 前缀（MLA 版 chunked prefill）。
        返回 out(1,S,d)、窗口 ckv(S,d_c)、窗口 krope(S,n_h,d_hR)。
        """
        c = self.cfg
        ckv, knope, v, qnope, qrope_pre, krope_pre = self._project(h)
        qfull, krope = self._rope_qk(qnope, qrope_pre, krope_pre, positions)

        t = prefix_ckv.shape[0]
        if t > 0:
            knope_p = F.linear(prefix_ckv, self.w_uk).view(t, c.n_h, c.d_h)
            v_p = F.linear(prefix_ckv, self.w_uv).view(t, c.n_h, c.d_h)
            knope_all = torch.cat((knope_p, knope[0]), dim=0)
            v_all = torch.cat((v_p, v[0]), dim=0)
            krope_all = torch.cat((prefix_krope, krope[0]), dim=0)
        else:
            knope_all, v_all, krope_all = knope[0], v[0], krope[0]

        kfull = torch.cat((knope_all, krope_all), dim=-1)
        s = h.shape[1]
        sk = t + s
        qidx = torch.arange(s, device=h.device).view(s, 1)
        kidx = torch.arange(sk, device=h.device).view(1, sk)
        visible = kidx <= (t + qidx)
        scale = 1.0 / c.head_total ** 0.5
        from .attention import attention_masked
        o = attention_masked(qfull, kfull.unsqueeze(0), v_all.unsqueeze(0), visible, scale)
        o = o.reshape(1, s, c.n_h * c.d_h)
        return F.linear(o, self.w_o), ckv[0], krope[0]

    # ---------- DECODE（单序列单 token）----------
    def _build_cache(self, h_t: torch.Tensor, ckv_hist: torch.Tensor,
                     krope_hist: torch.Tensor, position: int):
        """计算当前 token 并拼接历史，返回 ckv_all (T+1,d_c), krope_all (T+1,n_h,d_hR)。"""
        ckv_t, krope_t = self._cache_token(h_t, position)
        ckv_all = torch.cat((ckv_hist, ckv_t.unsqueeze(0)), dim=0)
        krope_all = torch.cat((krope_hist, krope_t.unsqueeze(0)), dim=0)
        return ckv_all, krope_all

    def decode_naive(self, h_t: torch.Tensor, ckv_hist: torch.Tensor,
                     krope_hist: torch.Tensor, position: int) -> torch.Tensor:
        """h_t: (d,)；历史 cache 不含当前。"""
        c = self.cfg
        ckv_all, krope_all = self._build_cache(h_t, ckv_hist, krope_hist, position)
        knope_all = F.linear(ckv_all, self.w_uk).view(-1, c.n_h, c.d_h)
        v_all = F.linear(ckv_all, self.w_uv).view(-1, c.n_h, c.d_h)
        qfull, _, _ = self._q_single(h_t, position)
        # 单 query 注意力：(n_h,head_total) x (T+1,n_h,head_total)
        scale = 1.0 / c.head_total ** 0.5
        kfull = torch.cat((knope_all, krope_all), dim=-1)
        scores = torch.einsum("hd,thd->ht", qfull, kfull) * scale
        probs = torch.softmax(scores, dim=-1)
        o = torch.einsum("ht,thd->hd", probs, v_all)
        return F.linear(o.reshape(-1), self.w_o)

    def decode_absorbed(self, h_t: torch.Tensor, ckv_hist: torch.Tensor,
                        krope_hist: torch.Tensor, position: int) -> torch.Tensor:
        """权重吸收：在 d_c 潜变量空间完成注意力。"""
        c = self.cfg
        ckv_all, krope_all = self._build_cache(h_t, ckv_hist, krope_hist, position)
        _, qnope, qrope = self._q_single(h_t, position)

        # q 吸收 W_UK：q_abs_h = W_UK_h^T q_nope_h，shape (n_h,d_c)
        w_uk = self.w_uk.view(c.n_h, c.d_h, c.d_c)
        q_abs = torch.einsum("hdc,hd->hc", w_uk, qnope)

        scale = 1.0 / c.head_total ** 0.5
        scores_nope = torch.einsum("hc,tc->ht", q_abs, ckv_all)
        scores_rope = torch.einsum("hR,thR->ht", qrope, krope_all)
        scores = (scores_nope + scores_rope) * scale
        probs = torch.softmax(scores, dim=-1)

        # v 吸收：先在潜变量空间加权，再 W_UV
        c_weighted = torch.einsum("ht,tc->hc", probs, ckv_all)  # (n_h,d_c)
        w_uv = self.w_uv.view(c.n_h, c.d_h, c.d_c)
        o = torch.einsum("hdc,hc->hd", w_uv, c_weighted)
        return F.linear(o.reshape(-1), self.w_o)
