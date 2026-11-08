# -*- coding: utf-8 -*-
"""
Mamba1 选择性 SSM 块（Selective State Space Model）。

一个 Mamba 块的完整数据流：
    x (B,L,d_model)
      -> in_proj 同时产出主干 x 与门控 z（各 d_inner）
      -> x 经过因果 depthwise 短卷积 + SiLU
      -> 由 x 数据相关地生成 Δ（步长）、B、C（选择性：参数随输入变化）
      -> 离散化：dA = exp(Δ·A)，dB = Δ·B
      -> 时变递推（用并行 scan）：h_t = dA_t·h_{t-1} + dB_t·x_t
      -> y = Σ_N(h·C) + D·x（D 为 skip 连接）
      -> 乘性门控 y·SiLU(z)
      -> out_proj 回到 d_model

数学门槛对应：L1（连续→离散化）、L2（时变递推与并行 scan）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CausalConv1d
from ..kernels.scan import scan, naive_scan


def selective_ssm_core(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                       B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                       use_parallel: bool = True):
    """
    选择性 SSM 的核心计算（离散化 + 扫描），单独拆出便于数值测试。

    x  : (B, L, d_inner)  卷积激活后的主干
    dt : (B, L, d_inner)  数据相关步长 Δ（已过 softplus）
    A  : (d_inner, N)     连续状态矩阵（负数）
    B  : (B, L, N)        数据相关输入矩阵
    C  : (B, L, N)        数据相关输出矩阵
    D  : (d_inner,)       skip 参数
    return: (B, L, d_inner)
    """
    bsz, L, di = x.shape
    N = A.shape[-1]

    # 离散化：
    # dA = exp(Δ·A)，形状 (B, L, d_inner, N)
    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    # dB = Δ·B（ZOH 的一阶近似），形状 (B, L, d_inner, N)
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)
    # 递推的"加项"：dB_t · x_t
    bterm = dB * x.unsqueeze(-1)  # (B, L, d_inner, N)

    # scan 要求序列在最后一维，这里把 (B,L,d_inner,N) 变为 (B,d_inner,N,L)
    a_flat = dA.permute(0, 2, 3, 1)
    b_flat = bterm.permute(0, 2, 3, 1)
    if use_parallel:
        h_flat = scan(a_flat, b_flat, block=128)
    else:
        h_flat = naive_scan(a_flat, b_flat)
    h = h_flat.permute(0, 3, 1, 2)  # (B, L, d_inner, N)

    # 输出：对状态维 N 求和，加上 skip 项 D·x
    y = (h * C.unsqueeze(2)).sum(dim=-1)  # (B, L, d_inner)
    y = y + D.unsqueeze(0).unsqueeze(0) * x
    # 返回所有位置的隐状态 h，调用方取 h[:, -1] 作为块末状态（decode 续接用）
    return y, h


class Mamba1Block(nn.Module):
    """完整的 Mamba1 块（含 in/out 投影、卷积、门控）。"""

    def __init__(self, d_model: int, d_inner: int, d_state: int = 16,
                 conv_kernel: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state

        # 输入投影：一次性产出主干与门控 z
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        # 因果短卷积
        self.conv = CausalConv1d(d_inner, conv_kernel)
        self.conv_kernel = conv_kernel

        # 数据相关投影：x -> Δ、B、C
        # 教学简化：Δ 直接投到 d_inner（真实实现走 dt_rank 再展开，数值等价）
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        # 连续参数 A：以 A_log 存储，A = -exp(A_log)，初始为负值
        A = torch.repeat_interleave(torch.arange(1, d_state + 1, dtype=torch.float32)
                                    .unsqueeze(0), d_inner, dim=0)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, N)
        # skip 参数 D
        self.D = nn.Parameter(torch.ones(d_inner))

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # 让初始 Δ 落在合理范围（softplus 后约为正）
        nn.init.zeros_(self.dt_proj.bias)

    def _ssm_params(self, x_act: torch.Tensor):
        """由卷积激活后的主干计算 Δ、B、C。"""
        # softplus 保证 Δ>0；Mamba 用 softplus 作为步长的正约束
        dt = F.softplus(self.dt_proj(x_act))
        B = self.B_proj(x_act)
        C = self.C_proj(x_act)
        return dt, B, C

    def forward(self, x: torch.Tensor, state: dict = None):
        """
        整段前向（prefill）。
        x     : (B,L,d_model)
        state : 可选初始状态 dict {"h","conv"}。教学版 prefill 暂不支持
                从非零状态续接（本函数体内未使用 state）；传进来会直接
                raise，而不是静默忽略后让调用方误以为续接生效。
        返回：(输出 (B,L,d_model), 新状态 dict {"h","conv"})，与 step 结构一致。
        """
        if state is not None:
            raise NotImplementedError(
                "Mamba2Layer.forward 暂不接受初始 state（教学版 prefill "
                "默认从零）；需要续接请用 step()，或在此接入 S0/conv 窗。")
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        x_act = F.silu(self.conv(x_part))
        dt, B, C = self._ssm_params(x_act)
        A = -torch.exp(self.A_log)  # 连续状态矩阵（负）
        y, h = selective_ssm_core(x_act, dt, A, B, C, self.D, use_parallel=True)
        y = y * F.silu(z)

        # 卷积窗口：卷积输入 x_part 的最后 conv_kernel-1 个位置，布局 (B,d_inner,k-1)
        kc = self.conv_kernel - 1
        xp = x_part.transpose(1, 2)
        if x_part.shape[1] >= kc:
            conv_window = xp[:, :, -kc:]
        else:
            conv_window = F.pad(xp, (kc - x_part.shape[1], 0))
        new_state = {"h": h[:, -1], "conv": conv_window}
        return self.out_proj(y), new_state

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict):
        """
        decode 单步：x_t (B,d_model)，state 含 h 与 conv_state。
        return: (输出 (B,d_model), 新 state)
        """
        xz = self.in_proj(x_t.unsqueeze(1))
        x_part, z = xz.chunk(2, dim=-1)
        # 卷积单步
        conv_out, new_conv = self.conv.step(x_part.squeeze(1), state["conv"])
        x_act = F.silu(conv_out)
        dt, B, C = self._ssm_params(x_act.unsqueeze(1))
        A = -torch.exp(self.A_log)

        # 单步离散化与递推（直接算，不做扫描）
        dA = torch.exp(dt.squeeze(1).unsqueeze(-1) * A.unsqueeze(0))  # (B,di,N)
        dB = dt.squeeze(1).unsqueeze(-1) * B.squeeze(1).unsqueeze(1)  # (B,di,N)
        h = dA * state["h"] + dB * x_act.squeeze(1).unsqueeze(-1)
        y = (h * C.squeeze(1).unsqueeze(1)).sum(-1) + self.D * x_act.squeeze(1)
        y = y * F.silu(z.squeeze(1))
        new_state = {"h": h, "conv": new_conv}
        return self.out_proj(y), new_state

    def init_state(self, batch: int, device, dtype=torch.float32):
        """初始化解码状态：隐状态 h 与卷积窗口。"""
        return {
            "h": torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype),
            "conv": torch.zeros(batch, self.d_inner, self.conv_kernel - 1, device=device,
                                dtype=dtype),
        }
