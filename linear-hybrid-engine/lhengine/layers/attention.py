# -*- coding: utf-8 -*-
"""
滑动窗口注意力（Sliding Window Attention, SWA）。

Hybrid 模型中，SWA 层负责"局部精确回忆"，与线性循环层（负责长程压缩）互补：
    - 因果 + 窗口：位置 t 只关注 [t-window+1, t] 范围内的 token
    - 支持 GQA：查询头数可多于 KV 头数（n_kv_heads）
    - prefill 用 PyTorch SDPA 加滑窗掩码；decode 维护固定大小的窗口 KV

注意：SWA 是本项目里唯一保留"真实 KV"的层，但其 KV 长度由窗口固定，
不随总上下文增长——这是它与全局注意力在显存上的本质区别。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sliding_window_mask(L: int, window: int, device) -> torch.Tensor:
    """
    构造因果滑动窗口掩码（加在 attention 分数上）。
    允许条件：j<=i 且 j > i-window，即 0 <= i-j < window。
    返回 (L, L)，允许处为 0、禁止处为 -inf。
    """
    i = torch.arange(L, device=device).reshape(L, 1)
    j = torch.arange(L, device=device).reshape(1, L)
    diff = i - j
    allowed = (diff >= 0) & (diff < window)
    mask = torch.zeros(L, L, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class SWABlock(nn.Module):
    """滑动窗口因果注意力块（含 GQA 与输出投影）。"""

    def __init__(self, d_model: int, n_heads: int, d_head: int = 64,
                 n_kv_heads: int = None, window: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.d_head = d_head
        self.window = window
        # Q 走 n_heads，K/V 走 n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def _repeat_kv(self, t: torch.Tensor):
        """把 KV 头复制到与查询头数一致（GQA）。t: (b,nkv,L,hd)。"""
        if self.n_kv_heads == self.n_heads:
            return t
        rep = self.n_heads // self.n_kv_heads
        t = t.repeat_interleave(rep, dim=1)
        return t

    def forward(self, x: torch.Tensor, state=None, mode: str = "window"):
        """
        整段前向（prefill）。
        返回 (输出 (b,L,d_model), state dict {"k","v","length"})，与 step 结构一致。
        """
        b, L, _ = x.shape
        q = self.q_proj(x).reshape(b, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(b, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(b, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        mask = sliding_window_mask(L, self.window, x.device)
        # SDPA：加性掩码（因果滑窗）
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0))
        y = y.transpose(1, 2).reshape(b, L, self.n_heads * self.d_head)

        # 固定长度窗口：取最后 window 个真实 KV，不足 window 时在"前面"补零，
        # 并用 length 记录真实 KV 数量（真实 KV 始终位于窗口末尾，最新在最后）。
        kc = k[:, :, -self.window:].contiguous()
        vc = v[:, :, -self.window:].contiguous()
        length = min(L, self.window)
        if L < self.window:
            kc = F.pad(kc, (0, 0, self.window - L, 0))
            vc = F.pad(vc, (0, 0, self.window - L, 0))
        new_state = {"k": kc, "v": vc, "length": length}
        return self.out_proj(y), new_state

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state):
        """
        decode 单步：state 为 dict {"k","v","length"}。
        关键：窗口里未被真实 KV 填充的位置必须在 attention 中 mask 掉，
        否则零向量 key 会以 0 分参与 softmax，稀释真实 KV 的注意力，
        与 prefill（这些位置被 -inf 掩码）不一致。
        """
        b = x_t.shape[0]
        q = self.q_proj(x_t).reshape(b, self.n_heads, self.d_head).unsqueeze(2)
        k = self.k_proj(x_t).reshape(b, self.n_kv_heads, self.d_head).unsqueeze(2)
        v = self.v_proj(x_t).reshape(b, self.n_kv_heads, self.d_head).unsqueeze(2)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        # 滑入新 KV，保持窗口长度（真实 KV 在末尾）
        k_all = torch.cat([state["k"], k], dim=2)[:, :, -self.window:]
        v_all = torch.cat([state["v"], v], dim=2)[:, :, -self.window:]
        new_length = min(state["length"] + 1, self.window)

        # 构造窗口位置的加性掩码：最后 new_length 个有效，前面 -inf
        pos_mask = torch.zeros(self.window, device=x_t.device, dtype=q.dtype)
        if new_length < self.window:
            pos_mask[:self.window - new_length] = float("-inf")
        attn_mask = pos_mask.reshape(1, 1, 1, self.window)
        y = F.scaled_dot_product_attention(
            q, k_all, v_all, attn_mask=attn_mask, is_causal=False)
        y = y.reshape(b, self.n_heads * self.d_head)
        new_state = {"k": k_all, "v": v_all, "length": new_length}
        return self.out_proj(y), new_state

    def init_state(self, batch: int, device, dtype=torch.float32):
        """初始化空窗口状态（length=0）。"""
        zero = torch.zeros(batch, self.n_heads, self.window, self.d_head,
                           device=device, dtype=dtype)
        return {"k": zero, "v": zero.clone(), "length": 0}
