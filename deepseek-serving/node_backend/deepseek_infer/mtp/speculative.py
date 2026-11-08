"""MTP 投机解码：单链/树状草稿与验证（贪心 + 拒绝采样）。

- 贪心验证：target argmax 与 draft 一致即接受，数学上对贪心解码无损；
- 拒绝采样：标准 speculative sampling，接受概率 min(1, p/q)，拒绝则从
  norm(max(0, p-q)) 重采样，数学上对随机采样无损且不改变 target 分布。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .mtp_head import MTPHead


# ---------------- 单链草稿 ----------------
@torch.no_grad()
def draft_chain(head: MTPHead, h_t: torch.Tensor, token_next: torch.Tensor,
                n_draft: int, greedy_draft: bool = True, temperature: float = 1.0
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 draft tokens (n_draft,) 与最后一步 hidden (1,1,d)。"""
    from ..engine.sampling import greedy, sample
    h = h_t
    tok = token_next
    toks = []
    for _ in range(n_draft):
        h, logits = head(h, tok)
        tok = greedy(logits[0, 0]) if greedy_draft else sample(logits[0, 0], temperature)
        toks.append(tok)
    return torch.stack(toks), h


# ---------------- 验证纯函数 ----------------
def accept_greedy(target_pred: torch.Tensor, drafts: torch.Tensor) -> int:
    """target_pred/drafts: (n,)。返回从头连续接受数。"""
    match = target_pred.to(drafts.device) == drafts
    false_idx = (~match).nonzero()
    return int(false_idx[0].item()) if false_idx.numel() > 0 else match.shape[0]


def rejection_accept_one(p: torch.Tensor, q: torch.Tensor, x: int,
                         r: float) -> tuple[bool, torch.Tensor | None]:
    """单位置拒绝采样。p/q 为该位置完整分布。返回 (是否接受, 重采样token或None)。"""
    px, qx = p[x].item(), q[x].item()
    if qx <= 0 or r < min(1.0, px / qx):
        return True, None
    # 从 norm(max(0, p-q)) 重采样
    adj = (p - q).clamp_min(0.0)
    adj = adj / adj.sum()
    return False, torch.multinomial(adj, 1)[0]


@torch.no_grad()
def target_probs_for(model, prefix_tokens, n_positions, temperature):
    """target 一次前向，取前 n_positions 个验证位置的分布。

    prefix_tokens: 包含起点 token_next 的完整序列（不含草稿）。
    返回 probs (n_positions, V)。
    """
    logits, _ = model.forward_prefill(prefix_tokens)
    probs = F.softmax(logits[0] / temperature, dim=-1)
    return probs[-n_positions:]


# ---------------- 树状草稿 ----------------
@dataclass
class Tree:
    tokens: torch.Tensor          # (N,)
    parents: torch.Tensor         # (N,)，-1 表示挂在虚拟根
    depths: torch.Tensor          # (N,)
    cum_logprob: torch.Tensor     # (N,)

    def paths(self) -> list[list[int]]:
        n = self.tokens.shape[0]
        result = []
        for leaf in range(n):
            if not (self.parents == leaf).any():  # 叶子节点
                path, cur = [], leaf
                while cur != -1:
                    path.append(cur)
                    cur = int(self.parents[cur].item())
                result.append(list(reversed(path)))
        return result


@torch.no_grad()
def draft_tree(head: MTPHead, h_t: torch.Tensor, token_next: torch.Tensor,
               width: int = 2, budget: int = 8, temperature: float = 1.0
               ) -> Tree:
    """best-first 动态树：每次扩展累计对数概率最高的叶子，加 width 个候选子节点。"""
    # 叶子：(parent_id, hidden, token, depth, cumlp)
    leaves = [(-1, h_t, token_next, 0, 0.0)]
    tokens, parents, depths, cumlps = [], [], [], []
    while len(tokens) < budget and leaves:
        leaves.sort(key=lambda z: z[4], reverse=True)
        pid, h_parent, tok_parent, depth, cumlp = leaves.pop(0)
        h, logits = head(h_parent, tok_parent)
        logp = F.log_softmax(logits[0, 0] / temperature, dim=-1)
        n_child = min(width, budget - len(tokens))
        topv, topi = torch.topk(logp, n_child)
        for v, t in zip(topv.tolist(), topi.tolist()):
            nid = len(tokens)
            tokens.append(t)
            parents.append(pid)
            depths.append(depth + 1)
            cumlps.append(cumlp + v)
            leaves.append((nid, h, torch.tensor(t, device=head.head_weight.device),
                           depth + 1, cumlp + v))
    dev = head.head_weight.device
    return Tree(
        tokens=torch.tensor(tokens, device=dev),
        parents=torch.tensor(parents, device=dev, dtype=torch.long),
        depths=torch.tensor(depths, device=dev, dtype=torch.long),
        cum_logprob=torch.tensor(cumlps, device=dev),
    )


def verify_tree_greedy(accept_depth_by_path: dict[int, int], tree: Tree,
                       target_bonus_by_path: dict[int, int]) -> tuple[list[int], int]:
    """根据每条路径的 target 接受深度，选深度最大（同深度累计概率最大）的路径。

    返回 (接受的节点 id 序列, bonus token)。
    """
    paths = tree.paths()
    best = None
    for pi, path in enumerate(paths):
        d = accept_depth_by_path[pi]
        accepted_nodes = path[:d]
        score = (tree.cum_logprob[accepted_nodes].sum().item() if d > 0 else -1e30)
        key = (d, score)
        if best is None or key > best[0]:
            best = (key, accepted_nodes, target_bonus_by_path[pi])
    _, nodes, bonus = best
    return nodes, bonus
