# -*- coding: utf-8 -*-
"""
Mamba2 块：多头 SSD（Structured State Duality）。

SSD 对偶（本项目数学枢纽 L3）：
    递推视图（逐 token）：
        S_t = dA_t · S_{t-1} + B_t ⊗ X_t      （S 是 N×P 的状态矩阵）
        y_t = C_t · S_t
    矩阵视图（整段）：
        Y = (M ⊙ C·Bᵀ) · X，M[i,j] = ∏_{k=j+1}^{i} dA_k
    两者严格等价。本文件同时实现：
        ssd_recurrent   循环递推（decode 用，也是数值参考）
        ssd_chunkwise   分块并行（prefill 用，块内 matmul + 块间状态递推）

Mamba2 相对 Mamba1 的关键简化：Δ 从"每维一个"改为"每头一个标量"，
    于是 dA 形状为 (L, H, N)，整个 SSM 变成标准的多头线性注意力 + 衰减。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CausalConv1d


def ssd_recurrent(X: torch.Tensor, dA: torch.Tensor, B: torch.Tensor,
                  C: torch.Tensor, S0: torch.Tensor = None):
    """
    循环版 SSD。

    X  : (b, L, H, P)
    dA : (b, L, H, N)  每步衰减（已 exp）
    B  : (b, L, H, N)
    C  : (b, L, H, N)
    S0 : (b, H, N, P)  初始状态，None 为零
    return: (y (b,L,H,P), final_state (b,H,N,P))
    """
    b, L, H, P = X.shape
    N = dA.shape[-1]
    S = torch.zeros(b, H, N, P, device=X.device, dtype=X.dtype) if S0 is None else S0
    outs = []
    for t in range(L):
        # 状态更新：先衰减旧状态，再外积写入当前 (B, X)
        S = dA[:, t].unsqueeze(-1) * S + torch.einsum("bhn,bhp->bhnp", B[:, t], X[:, t])
        # 读取：C 与状态矩阵相乘
        y = torch.einsum("bhn,bhnp->bhp", C[:, t], S)
        outs.append(y)
    y = torch.stack(outs, dim=1)
    return y, S


def ssd_chunkwise(X: torch.Tensor, log_dA: torch.Tensor, B: torch.Tensor,
                  C: torch.Tensor, chunk: int = 64, S0: torch.Tensor = None):
    """
    分块并行 SSD（chunkwise）。

    log_dA : (b, L, H, N) 衰减的对数（即 Δ·A），用于稳定地构造 decay 矩阵
    其余同 ssd_recurrent。
    return: (y (b,L,H,P), final_state (b,H,N,P))

    算法（对每个块，块间循环、块内并行）：
        cum   = 块内 cumsum(log_dA)
        M[i,j]= exp(cum[i]-cum[j])（j<=i），即块内 decay 矩阵
        块内输出 Y_intra = (M ⊙ C Bᵀ) X
        块间输出 Y_inter = C·(exp(cum)·S)
        块末状态 S' = exp(cum[-1])·S + (exp(cum[-1]-cum)⊙B) X
    """
    b, L, H, P = X.shape
    N = log_dA.shape[-1]
    nb = (L + chunk - 1) // chunk
    pad = nb * chunk - L
    # 在序列维末尾 pad 到整块（log_dA pad 0，X pad 0，B/C pad 0）
    if pad:
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        log_dA = F.pad(log_dA, (0, 0, 0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))

    # 拆出"块号 q、块内位置 r"
    X = X.reshape(b, nb, chunk, H, P)
    log_dA = log_dA.reshape(b, nb, chunk, H, N)
    B = B.reshape(b, nb, chunk, H, N)
    C = C.reshape(b, nb, chunk, H, N)

    # 块内沿位置的 cumsum
    cum = log_dA.cumsum(dim=2)                       # (b,q,r,H,N)
    # M[i,j] = exp(cum_i - cum_j)：在 r 维（dim=2）上做广播。
    # cum_i 需为 (b,q,r_i,1,H,N) -> cum.unsqueeze(3)
    # cum_j 需为 (b,q,1,r_j,H,N) -> cum.unsqueeze(2)
    pos = torch.arange(chunk, device=X.device)
    causal_bool = (pos[:, None] >= pos[None, :])          # (c,c)，j<=i 为 True
    diff = cum.unsqueeze(3) - cum.unsqueeze(2)           # (b,q,r_i,r_j,H,N)
    # 关键：必须先把"未来位置" j>i 填成 -inf 再 exp，
    # 否则未来位置 exp(大正数)=inf，再乘 0 会得到 inf*0=NaN。
    diff = diff.masked_fill(~causal_bool.reshape(1, 1, chunk, chunk, 1, 1),
                            float("-inf"))
    M = diff.exp()                                       # (b,q,r_i,r_j,H,N)，未来位置为 0

    # 块内项 MCB[i,j,H] = Σ_n C[i,n]·M[i,j,n]·B[j,n]
    MCB = torch.einsum("bqihn,bqijhn,bqjhn->bqijh", C, M, B)
    # Y_intra[i,p] = Σ_j MCB[i,j]·X[j,p]
    Y_intra = torch.einsum("bqijh,bqjhp->bqihp", MCB, X)

    # 逐块传播跨块状态，计算 Y_inter 并更新 S
    S = torch.zeros(b, H, N, P, device=X.device, dtype=X.dtype) if S0 is None else S0
    Y_inter = []
    for q in range(nb):
        d = cum[:, q].exp()                          # 块首到 i 的衰减 (b,r,H,N)
        # y_inter = C·(d·S)
        yi = torch.einsum("brhn,brhn,bhnp->brhp", C[:, q], d, S)
        Y_inter.append(yi)
        # 块末状态：整体衰减 + 块内"反向衰减"外积累加
        chunk_decay = cum[:, q, -1].exp()           # (b,H,N)
        end_decay = (cum[:, q, -1:] - cum[:, q]).exp()  # (b,r,H,N)
        S_intra = torch.einsum("brhn,brhn,brhp->bhnp", end_decay, B[:, q], X[:, q])
        S = chunk_decay.unsqueeze(-1) * S + S_intra
    Y_inter = torch.stack(Y_inter, dim=1)            # (b,nb,r,H,P)

    Y = (Y_intra + Y_inter).reshape(b, nb * chunk, H, P)
    return Y[:, :L], S


class Mamba2Block(nn.Module):
    """完整的 Mamba2 块（含 in/out 投影、卷积、门控、多头 SSD）。"""

    def __init__(self, d_model: int, d_inner: int, n_heads: int,
                 d_state: int = 128, conv_kernel: int = 4, chunk: int = 64):
        super().__init__()
        assert d_inner % n_heads == 0
        self.d_model, self.d_inner = d_model, d_inner
        self.n_heads, self.d_state = n_heads, d_state
        self.d_head = d_inner // n_heads
        self.chunk = chunk

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = CausalConv1d(d_inner, conv_kernel)
        self.conv_kernel = conv_kernel

        # Mamba2：Δ 每头一个标量
        self.dt_proj = nn.Linear(d_inner, n_heads, bias=True)
        # B、C 为多头：d_inner -> H*N
        self.B_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)

        # 连续 A：(H, N)，A_log 存储，A=-exp(A_log)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(n_heads, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # 每头 skip 参数
        self.D = nn.Parameter(torch.ones(n_heads))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        nn.init.zeros_(self.dt_proj.bias)

    def _split_heads(self, t: torch.Tensor):
        """(b, L, H*N) -> (b, L, H, N)。"""
        b, L, _ = t.shape
        return t.reshape(b, L, self.n_heads, self.d_state)

    def _ssd_inputs(self, x_act: torch.Tensor):
        """由卷积激活后的主干构造 SSD 输入：X、log_dA、B、C。"""
        b, L, _ = x_act.shape
        dt = F.softplus(self.dt_proj(x_act))                 # (b,L,H)
        A = -torch.exp(self.A_log)                            # (H,N)
        log_dA = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (b,L,H,N)
        B = self._split_heads(self.B_proj(x_act))
        C = self._split_heads(self.C_proj(x_act))
        X = x_act.reshape(b, L, self.n_heads, self.d_head)
        return X, log_dA, B, C

    def forward(self, x: torch.Tensor, state=None,
                mode: str = "chunkwise"):
        """
        整段前向（prefill）。
        mode : "chunkwise"（并行，prefill 默认）或 "recurrent"（循环，参考/对齐）
        state: 可选初始状态，dict {"S","conv"}（decode 续接）或直接给 S 张量
        返回：(输出 (b,L,d_model), 新状态 dict {"S","conv"})，与 step 的状态结构一致。
        """
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        x_act = F.silu(self.conv(x_part))
        X, log_dA, B, C = self._ssd_inputs(x_act)

        # 初始状态若是 dict，取出其中的 SSD 状态 S
        S0 = state["S"] if isinstance(state, dict) else state
        if mode == "recurrent":
            dA = log_dA.exp()
            y, S = ssd_recurrent(X, dA, B, C, S0=S0)
        else:
            y, S = ssd_chunkwise(X, log_dA, B, C, chunk=self.chunk, S0=S0)
        # 每头 skip 参数 D 广播到 (b,L,H,P)，再拼回 d_inner
        bsz, L = x.shape[0], x.shape[1]
        skip = self.D.view(1, 1, self.n_heads, 1)
        y = (y + skip * X).reshape(bsz, L, self.d_inner)
        y = y * F.silu(z)

        # 卷积窗口：卷积输入 x_part 的最后 conv_kernel-1 个位置，供 decode 续接。
        # 布局与 init_state 一致：(b, d_inner, conv_kernel-1)。
        kc = self.conv_kernel - 1
        xp = x_part.transpose(1, 2)                       # (b,d_inner,L)
        if x_part.shape[1] >= kc:
            conv_window = xp[:, :, -kc:]
        else:                                             # 序列比窗口短，左侧补零
            conv_window = F.pad(xp, (kc - x_part.shape[1], 0))
        new_state = {"S": S, "conv": conv_window}
        return self.out_proj(y), new_state

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict):
        """
        decode 单步：x_t (b,d_model)，state 含 S（SSD 状态）与 conv_state。
        return: (输出 (b,d_model), 新 state)
        """
        xz = self.in_proj(x_t.unsqueeze(1))
        x_part, z = xz.chunk(2, dim=-1)
        conv_out, new_conv = self.conv.step(x_part.squeeze(1), state["conv"])
        x_act = F.silu(conv_out)
        X, log_dA, B, C = self._ssd_inputs(x_act.unsqueeze(1))

        # 单步状态更新（取 t=0）
        S = log_dA[:, 0].exp().unsqueeze(-1) * state["S"] + torch.einsum(
            "bhn,bhp->bhnp", B[:, 0], X[:, 0])
        y = torch.einsum("bhn,bhnp->bhp", C[:, 0], S)
        # 每头 skip 后拼回 d_inner
        bsz = x_t.shape[0]
        y = (y + self.D.view(1, self.n_heads, 1) * X[:, 0]).reshape(bsz, self.d_inner)
        y = y * F.silu(z.squeeze(1))
        return self.out_proj(y), {"S": S, "conv": new_conv}

    def init_state(self, batch: int, device, dtype=torch.float32):
        """初始化解码状态：SSD 状态矩阵 S 与卷积窗口。"""
        return {
            "S": torch.zeros(batch, self.n_heads, self.d_state, self.d_head,
                             device=device, dtype=dtype),
            "conv": torch.zeros(batch, self.d_inner, self.conv_kernel - 1,
                                device=device, dtype=dtype),
        }
