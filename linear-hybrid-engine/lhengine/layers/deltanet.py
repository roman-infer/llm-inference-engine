# -*- coding: utf-8 -*-
"""
DeltaNet / Gated DeltaNet 层（delta rule 线性注意力）。

delta rule（数学门槛 L4）：
    不像线性注意力那样直接把 v 累加进记忆，而是先问"记忆对当前 key 已经预测了什么"，
    只把"预测残差"写进去：
        v̄ = α·W·k
        r = v − v̄
        W = α·W + β·r⊗k
        y = W·q
    这等价于对在线回归损失 ½‖W·k − v‖² 做一步 SGD，
    其中 I−β·k·kᵀ 是 Householder 式投影，先擦除 key 方向旧内容再写新值。
    α 控制全局遗忘（gated），β 控制定向写入强度。

chunkwise 并行（数学门槛 L5，本文件最难）：
    用 WY 表示把块内一串 Householder 乘积压缩成 I−WᵀK，
    用 UT 变换把递推变成一个（下三角）线性系统的解，最终全部化为 matmul：
        S' = γᶜ·S + Rᵀ·K̃，其中 R = Ũ − W̃·Sᵀ
        O  = Q̃·Sᵀ + (Q·Kᵀ ⊙ M)·R
    无门控时 α=1、γ=1 即退化为普通 DeltaNet。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def delta_recurrent(K: torch.Tensor, V: torch.Tensor, Q: torch.Tensor,
                    alpha: torch.Tensor, beta: torch.Tensor,
                    S0: torch.Tensor = None):
    """
    循环版（gated delta rule）。

    K, Q : (b, L, H, Kdim)
    V    : (b, L, H, P)
    alpha: (b, L, H)  遗忘门（每步）
    beta : (b, L, H)  写入强度
    S0   : (b, H, P, Kdim) 初始状态
    return: (y (b,L,H,P), S (b,H,P,Kdim))
    """
    b, L, H, Kdim = K.shape
    P = V.shape[-1]
    S = torch.zeros(b, H, P, Kdim, device=K.device, dtype=K.dtype) if S0 is None else S0
    outs = []
    for t in range(L):
        a = alpha[:, t]   # (b,H) 遗忘门
        bt = beta[:, t]   # (b,H) 写入强度
        # 记忆对当前 key 的预测，再乘遗忘门（标量）
        vbar = torch.einsum("bhpk,bhk->bhp", S, K[:, t])
        vbar = a.view(b, H, 1) * vbar
        r = V[:, t] - vbar
        # 状态更新：整体衰减 a·S + 残差外积 β·r⊗k
        S = a.view(b, H, 1, 1) * S + bt.view(b, H, 1, 1) * torch.einsum(
            "bhp,bhk->bhpk", r, K[:, t])
        y = torch.einsum("bhpk,bhk->bhp", S, Q[:, t])
        outs.append(y)
    return torch.stack(outs, dim=1), S


def _ut_transform(K: torch.Tensor, V: torch.Tensor, beta: torch.Tensor,
                  log_gamma: torch.Tensor):
    """
    UT 变换，返回 (W, U)。

    K    : (b, H, c, Kdim)
    V    : (b, H, c, P)
    beta : (b, H, c)
    gamma: (b, H, c) 块内累积衰减 γ^r（gated 时为 cumprod(α)，ungated 时全 1）

    数学上（见本项目 docs/math_notes）：
      - W 是"标准 delta WY"的 W（不含衰减 Γ）：W = T_ung·K，
        外层再由调用方乘 γ 得到 overleftarrow W；
      - U 是"衰减感知"的 tilde U：U = T_g·V（gated 含 Γ，ungated 时 γ=1 自动退化）。
    两者必须分开，不能都用同一个含 Γ 的 T，否则 W 会被重复衰减。
    """
    b, H, c, Kdim = K.shape
    pos = torch.arange(c, device=K.device)
    eye = torch.eye(c, device=K.device, dtype=K.dtype).reshape(1, 1, c, c)
    diag_beta = torch.diag_embed(beta)                       # (b,H,c,c)
    # 块内 token 间 key 内积 K K^T：(i,j) = k_i·k_j
    KK = torch.einsum("bhik,bhjk->bhij", K, K)               # (b,H,c,c)
    # 严格下三角布尔（i>j），用布尔而非 0/1 相乘，避免任何潜在的 0·inf
    lower_bool = (pos[:, None] > pos[None, :])               # (c,c)

    # ---- 标准（ungated）W：L_ung = strictLower(diag(β)·K K^T) ----
    # 通用三角求解在 fp32 中进行：cuSOLVER / CPU LU 不支持 half，且
    # 单位下三角系统对数值敏感；解完再投回原 dtype。
    solve_dt = torch.float32
    L_ung = (beta.unsqueeze(3) * KK).masked_fill(~lower_bool.reshape(1, 1, c, c), 0.0)
    T_ung = torch.linalg.solve(
        (eye + L_ung).to(solve_dt), diag_beta.to(solve_dt)
    ).to(K.dtype)
    W = torch.einsum("bhij,bhjk->bhik", T_ung, K)

    # ---- 衰减感知 tilde U：L_g = strictLower(diag(β)·(Γ ⊙ K K^T)) ----
    # Γ[i,j] = γ_i/γ_j = exp(log_gamma_i - log_gamma_j)（i>j 时 ≤1）。
    # clamp(max=0) 防止 upper 区域 exp 爆炸，随后被 causal mask 掉；
    # 用 log 域差代替 cumprod 比值，避免小 α 下 γ 下溢成 0 后 0/0=NaN。
    log_Gamma = log_gamma.unsqueeze(3) - log_gamma.unsqueeze(2)
    Gamma = torch.exp(log_Gamma.clamp(max=0.0))
    L_g = (beta.unsqueeze(3) * Gamma * KK).masked_fill(
        ~lower_bool.reshape(1, 1, c, c), 0.0)
    T_g = torch.linalg.solve(
        (eye + L_g).to(solve_dt), diag_beta.to(solve_dt)
    ).to(K.dtype)
    U = torch.einsum("bhij,bhjp->bhip", T_g, V)
    return W, U


def delta_chunkwise(K: torch.Tensor, V: torch.Tensor, Q: torch.Tensor,
                    alpha: torch.Tensor, beta: torch.Tensor,
                    chunk: int = 64, S0: torch.Tensor = None):
    """
    分块并行 gated delta rule（WY/UT）。

    输入维度同 delta_recurrent，序列在 dim=1。
    return: (y (b,L,H,P), S (b,H,P,Kdim))
    """
    b, L, H, Kdim = K.shape
    P = V.shape[-1]
    nb = (L + chunk - 1) // chunk
    pad = nb * chunk - L
    # 序列维 pad（alpha pad 1，其余 pad 0）
    if pad:
        K = F.pad(K, (0, 0, 0, 0, 0, pad))
        V = F.pad(V, (0, 0, 0, 0, 0, pad))
        Q = F.pad(Q, (0, 0, 0, 0, 0, pad))
        alpha = F.pad(alpha, (0, 0, 0, pad), value=1.0)
        beta = F.pad(beta, (0, 0, 0, pad))

    S = torch.zeros(b, H, P, Kdim, device=K.device, dtype=K.dtype) if S0 is None else S0
    ys = []
    for q in range(nb):
        # 取当前块并转成 (b,H,c,...)
        Kc = K[:, q * chunk:(q + 1) * chunk].transpose(1, 2)
        Vc = V[:, q * chunk:(q + 1) * chunk].transpose(1, 2)
        Qc = Q[:, q * chunk:(q + 1) * chunk].transpose(1, 2)
        ac = alpha[:, q * chunk:(q + 1) * chunk].transpose(1, 2)
        bc = beta[:, q * chunk:(q + 1) * chunk].transpose(1, 2)

        # 块内累积衰减在 log 域维护：γ^r = ∏α 直接 cumprod 在小 α（门控
        # sigmoid 常落到 0.1~0.3）下会下溢成 0，使末位 γᶜ/γ^r 变成 0/0=NaN。
        # log_gamma = cumsum(log α)，所有比值改 exp(log_gamma_i - log_gamma_j)。
        log_gamma = torch.cumsum(torch.log(ac.clamp_min(1e-30)), dim=2)
        log_gammaC = log_gamma[:, :, -1]                       # (b,H)

        W, U = _ut_transform(Kc, Vc, bc, log_gamma)
        Wt = torch.exp(log_gamma).unsqueeze(-1) * W            # W̃ = γ^r·W
        # K̃ = γᶜ/γ^r·K = exp(log_gammaC - log_gamma)；末位差值为 0 → exp(0)=1
        Kt = torch.exp(log_gammaC.unsqueeze(-1) - log_gamma).unsqueeze(-1) * Kc
        Qt = torch.exp(log_gamma).unsqueeze(-1) * Qc           # Q̃ = γ^r·Q

        # R = Ũ − W̃·Sᵀ
        St = S.transpose(-1, -2)                            # (b,H,Kdim,P)
        R = U - torch.einsum("bhik,bhkp->bhip", Wt, St)     # (b,H,c,P)

        # 块内输出：overleftarrow Q·S^T + (Γ ⊙ Q·K^T)·R
        y_inter = torch.einsum("bhik,bhkp->bhip", Qt, St)
        QK = torch.einsum("bhik,bhjk->bhij", Qc, Kc)        # Q·K^T
        # Γ[r,i] = γ_r/γ_i = exp(log_gamma_r - log_gamma_i)；ungated 时 α=1，
        # log_gamma=0，Γ 退化为全 1。
        log_Gamma_out = log_gamma.unsqueeze(3) - log_gamma.unsqueeze(2)
        Gamma_out = torch.exp(log_Gamma_out.clamp(max=0.0))  # (b,H,c,c)
        pos = torch.arange(chunk, device=K.device)
        causal_bool = (pos[:, None] >= pos[None, :])
        Gamma_masked = Gamma_out.masked_fill(
            ~causal_bool.reshape(1, 1, chunk, chunk), 0.0)
        y_intra = torch.einsum("bhij,bhjp->bhip", Gamma_masked * QK, R)
        y = (y_inter + y_intra).transpose(1, 2)             # (b,c,H,P)
        ys.append(y)

        # 状态更新：γᶜ·S + Rᵀ·K̃（γᶜ = exp(log_gammaC)）
        gammaC_val = torch.exp(log_gammaC).view(b, H, 1, 1)
        S = gammaC_val * S + torch.einsum("bhcp,bhck->bhpk", R, Kt)

    y = torch.cat(ys, dim=1)
    return y[:, :L], S


class GatedDeltaNetBlock(nn.Module):
    """
    Gated DeltaNet 块（外层）。

    与 Mamba 块不同：GDN 没有短卷积，主干是 delta 线性层；
    q/k/v 由输入投影，α/β 由输入经过 sigmoid 产生。
    gated=False 时为纯 DeltaNet（α 恒为 1）。
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int = 64,
                 d_key: int = 64, gated: bool = True, chunk: int = 64):
        super().__init__()
        self.n_heads, self.d_head, self.d_key = n_heads, d_head, d_key
        self.gated, self.chunk = gated, chunk
        P = n_heads * d_head
        # 合并投影：一次性产出 q,k,v
        self.qkv_proj = nn.Linear(d_model, n_heads * (2 * d_key + d_head), bias=False)
        # 门控：α（遗忘）、β（写入）
        self.gate_proj = nn.Linear(d_model, 2 * n_heads, bias=True)
        self.out_proj = nn.Linear(P, d_model, bias=False)

    def _split(self, x: torch.Tensor):
        """生成多头 K、V、Q、α、β。"""
        b, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(
            [self.n_heads * self.d_key, self.n_heads * self.d_key,
             self.n_heads * self.d_head], dim=-1)
        K = k.reshape(b, L, self.n_heads, self.d_key)
        Q = q.reshape(b, L, self.n_heads, self.d_key)
        V = v.reshape(b, L, self.n_heads, self.d_head)
        # 论文明确：对 q、k 做 L2 归一化是训练/数值稳定的关键。
        # 归一化后 ||k||=1，delta 矩阵 (I-β·k k^T) 的特征值落在 (0,1)，状态不发散；
        # 否则 ||k||>1 时特征值绝对值可能 >1，状态指数爆炸、chunkwise 求解奇异。
        K = F.normalize(K, dim=-1)
        Q = F.normalize(Q, dim=-1)
        gates = torch.sigmoid(self.gate_proj(x))
        alpha, beta = gates.chunk(2, dim=-1)             # 各 (b,L,H)
        if not self.gated:
            alpha = torch.ones_like(alpha)
        return K, V, Q, alpha, beta

    def forward(self, x: torch.Tensor, state: torch.Tensor = None,
                mode: str = "chunkwise"):
        """整段前向，返回 (输出 (b,L,d_model), 状态 S)。"""
        K, V, Q, alpha, beta = self._split(x)
        if mode == "recurrent":
            y, S = delta_recurrent(K, V, Q, alpha, beta, S0=state)
        else:
            y, S = delta_chunkwise(K, V, Q, alpha, beta, chunk=self.chunk, S0=state)
        y = y.reshape(x.shape[0], x.shape[1], self.n_heads * self.d_head)
        return self.out_proj(y), S

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, S: torch.Tensor):
        """decode 单步：x_t (b,d_model)，S (b,H,P,K)。"""
        K, V, Q, alpha, beta = self._split(x_t.unsqueeze(1))
        # 直接用单元素递推
        a = alpha[:, 0].view(-1, self.n_heads, 1, 1)
        bt = beta[:, 0].view(-1, self.n_heads, 1, 1)
        vbar = torch.einsum("bhpk,bhk->bhp", S, K[:, 0])
        r = V[:, 0] - alpha[:, 0].view(-1, self.n_heads, 1) * vbar
        S = a * S + bt * torch.einsum("bhp,bhk->bhpk", r, K[:, 0])
        y = torch.einsum("bhpk,bhk->bhp", S, Q[:, 0])
        y = y.reshape(x_t.shape[0], self.n_heads * self.d_head)
        return self.out_proj(y), S

    def init_state(self, batch: int, device, dtype=torch.float32):
        """初始化 delta 状态矩阵 (b,H,P,K)。"""
        return torch.zeros(batch, self.n_heads, self.d_head, self.d_key,
                           device=device, dtype=dtype)
