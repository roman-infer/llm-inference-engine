# -*- coding: utf-8 -*-
"""
Kimi Delta Attention（KDA）层。

相对 Gated DeltaNet 的升级（数学门槛 L6）：
    遗忘门从"标量"（每头一个 α）细化为"通道向量"（每个 key 特征维一个 α）：
        v̄ = S·(α ⊙ k)          （α 是 Kdim 维向量，逐通道衰减）
        r = v − v̄
        S = S·diag(α) + β·r⊗k
        y = S·q
    标量门是通道门的特例：α = α_scalar·1 时退化为 GDN。
    通道门让模型能对记忆的不同特征子空间分别保留/遗忘，表达力更强。

关于 chunkwise：
    完整的 KDA 分块并行涉及"对角 + 低秩（DPLR）转移矩阵"与 Woodbury 恒等式，
    是 Kimi Linear 技术报告的进阶内容。本文件提供：
      kda_recurrent      严格递推（decode 用，数值基准）
      kda_block_loop     块内循环 + 块间状态（prefill 用，结果严格正确，
                         块内并行度低于完整 DPLR 内核）
    两者数值一致；完整 DPLR chunkwise 留作 CUDA 内核进阶（见 csrc 注释）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def kda_recurrent(K: torch.Tensor, V: torch.Tensor, Q: torch.Tensor,
                  alpha: torch.Tensor, beta: torch.Tensor, S0: torch.Tensor = None):
    """
    循环版 KDA。
    K, Q : (b, L, H, Kdim)
    V    : (b, L, H, P)
    alpha: (b, L, H, Kdim)  通道遗忘（每步每通道）
    beta : (b, L, H)        写入强度
    return: (y (b,L,H,P), S (b,H,P,Kdim))
    """
    b, L, H, Kdim = K.shape
    P = V.shape[-1]
    S = torch.zeros(b, H, P, Kdim, device=K.device, dtype=K.dtype) if S0 is None else S0
    outs = []
    for t in range(L):
        # 通道衰减后的 key：α ⊙ k
        k_dec = alpha[:, t] * K[:, t]                # (b,H,Kdim)
        vbar = torch.einsum("bhpk,bhk->bhp", S, k_dec)
        r = V[:, t] - vbar
        # S = S·diag(α) + β·r⊗k
        S = S * alpha[:, t].unsqueeze(2) + beta[:, t].view(b, H, 1, 1) * torch.einsum(
            "bhp,bhk->bhpk", r, K[:, t])
        y = torch.einsum("bhpk,bhk->bhp", S, Q[:, t])
        outs.append(y)
    return torch.stack(outs, dim=1), S


def kda_block_loop(K: torch.Tensor, V: torch.Tensor, Q: torch.Tensor,
                   alpha: torch.Tensor, beta: torch.Tensor, chunk: int = 64,
                   S0: torch.Tensor = None):
    """
    块内循环 + 块间状态的 KDA。
    块内严格按递推计算（结果与 kda_recurrent 完全一致），
    仅把"块间"的状态传播独立出来，便于与完整 DPLR chunkwise 对照替换。
    """
    b, L, H, Kdim = K.shape
    P = V.shape[-1]
    nb = (L + chunk - 1) // chunk
    S = torch.zeros(b, H, P, Kdim, device=K.device, dtype=K.dtype) if S0 is None else S0
    ys = []
    for q in range(nb):
        lo, hi = q * chunk, min((q + 1) * chunk, L)
        # 块内从当前状态 S 出发逐 token 递推
        Kc, Vc, Qc = K[:, lo:hi], V[:, lo:hi], Q[:, lo:hi]
        ac, bc = alpha[:, lo:hi], beta[:, lo:hi]
        cL = hi - lo
        outs = []
        for r in range(cL):
            k_dec = ac[:, r] * Kc[:, r]
            vbar = torch.einsum("bhpk,bhk->bhp", S, k_dec)
            rr = Vc[:, r] - vbar
            S = S * ac[:, r].unsqueeze(2) + bc[:, r].view(b, H, 1, 1) * torch.einsum(
                "bhp,bhk->bhpk", rr, Kc[:, r])
            outs.append(torch.einsum("bhpk,bhk->bhp", S, Qc[:, r]))
        ys.append(torch.stack(outs, dim=1))
    return torch.cat(ys, dim=1), S


class KDABlock(nn.Module):
    """Kimi Delta Attention 完整块（外层投影 + 通道门控）。"""

    def __init__(self, d_model: int, n_heads: int, d_head: int = 64,
                 d_key: int = 64, chunk: int = 64):
        super().__init__()
        self.n_heads, self.d_head, self.d_key = n_heads, d_head, d_key
        self.chunk = chunk
        P = n_heads * d_head
        self.qkv_proj = nn.Linear(d_model, n_heads * (2 * d_key + d_head), bias=False)
        # 通道遗忘门 α：投影到 H*Kdim；写入强度 β：投影到 H
        self.alpha_proj = nn.Linear(d_model, n_heads * d_key, bias=True)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(P, d_model, bias=False)

    def _split(self, x: torch.Tensor):
        b, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(
            [self.n_heads * self.d_key, self.n_heads * self.d_key,
             self.n_heads * self.d_head], dim=-1)
        K = k.reshape(b, L, self.n_heads, self.d_key)
        Q = q.reshape(b, L, self.n_heads, self.d_key)
        V = v.reshape(b, L, self.n_heads, self.d_head)
        # 通道 α（sigmoid 到 (0,1)），标量 β
        alpha = torch.sigmoid(self.alpha_proj(x)).reshape(b, L, self.n_heads, self.d_key)
        beta = torch.sigmoid(self.beta_proj(x))
        return K, V, Q, alpha, beta

    def forward(self, x: torch.Tensor, state=None, mode: str = "block"):
        """整段前向：mode 'block'（块内循环 prefill）或 'recurrent'。"""
        K, V, Q, alpha, beta = self._split(x)
        if mode == "recurrent":
            y, S = kda_recurrent(K, V, Q, alpha, beta, S0=state)
        else:
            y, S = kda_block_loop(K, V, Q, alpha, beta, chunk=self.chunk, S0=state)
        y = y.reshape(x.shape[0], x.shape[1], self.n_heads * self.d_head)
        return self.out_proj(y), S

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, S: torch.Tensor):
        """decode 单步：S (b,H,P,K)。"""
        K, V, Q, alpha, beta = self._split(x_t.unsqueeze(1))
        k_dec = alpha[:, 0] * K[:, 0]
        vbar = torch.einsum("bhpk,bhk->bhp", S, k_dec)
        r = V[:, 0] - vbar
        S = S * alpha[:, 0].unsqueeze(2) + beta[:, 0].view(-1, self.n_heads, 1, 1) * torch.einsum(
            "bhp,bhk->bhpk", r, K[:, 0])
        y = torch.einsum("bhpk,bhk->bhp", S, Q[:, 0])
        y = y.reshape(x_t.shape[0], self.n_heads * self.d_head)
        return self.out_proj(y), S

    def init_state(self, batch: int, device, dtype=torch.float32):
        return torch.zeros(batch, self.n_heads, self.d_head, self.d_key,
                           device=device, dtype=dtype)
