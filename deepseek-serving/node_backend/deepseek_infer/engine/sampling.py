"""采样：greedy / temperature / top-k / top-p。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(-1)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
           top_p: float = 1.0) -> torch.Tensor:
    """logits: (..., V)。"""
    if temperature <= 0:
        return greedy(logits)
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        v, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < v[..., -1:], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cs = torch.cumsum(sp, dim=-1)
        rm = cs - sp > top_p
        sp = sp.masked_fill(rm, 0.0)
        sp = sp / sp.sum(-1, keepdim=True)
        probs = torch.empty_like(probs).scatter_(-1, si, sp)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])
