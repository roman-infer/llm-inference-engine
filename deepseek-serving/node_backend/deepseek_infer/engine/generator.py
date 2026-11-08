"""生成器：普通自回归生成与 MTP 投机解码生成（贪心，无损）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..model.model import DeepSeekModel
from ..mtp.mtp_head import MTPHead
from ..mtp.speculative import accept_greedy, draft_chain
from .kv_cache import LatentKVCache
from .sampling import greedy


@dataclass
class MTPStats:
    rounds: int = 0
    accepted: list = field(default_factory=list)

    @property
    def mean_accept(self) -> float:
        return sum(self.accepted) / max(1, len(self.accepted))


class Generator:
    def __init__(self, model: DeepSeekModel, head: MTPHead | None = None,
                 cache: LatentKVCache | None = None, device="cuda"):
        self.model = model
        self.head = head
        self.cache = cache
        if cache is not None:
            model.cache = cache

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int) -> torch.Tensor:
        """普通 target 自回归（贪心）。prompt: (P,)。"""
        self.cache.reset()
        logits, _ = self.model.forward_prefill(prompt)
        out = list(prompt.tolist())
        position = prompt.shape[0]
        for i in range(max_new):
            nxt = greedy(logits[0, -1])
            out.append(int(nxt.item()))
            if i < max_new - 1:
                logits, _ = self.model.forward_decode(nxt, position)
                position += 1
        return torch.tensor(out, device=prompt.device)

    @torch.no_grad()
    def generate_mtp(self, prompt: torch.Tensor, max_new: int,
                     n_draft: int = 4) -> tuple[torch.Tensor, MTPStats]:
        """MTP 投机解码（贪心验证，无损）。"""
        self.cache.reset()
        stats = MTPStats()
        logits, _ = self.model.forward_prefill(prompt)
        P = prompt.shape[0]

        # 第一个新 token 由 target 正常产生并前向，建立 draft 起点
        nxt = greedy(logits[0, -1])
        logits_c, h_t = self.model.forward_continue(nxt.view(1), P)
        pos = P
        token_next = nxt
        produced = 1
        out = [int(nxt.item())]

        while produced < max_new:
            drafts, _ = draft_chain(self.head, h_t, token_next, n_draft)
            vlogits, _ = self.model.forward_continue(drafts, pos + 1)

            # target 各验证位置预测：at pos 用上轮 logits_c，at pos+1..pos+n-1 用 vlogits
            preds = [greedy(logits_c[0, -1])]
            for j in range(n_draft - 1):
                preds.append(greedy(vlogits[0, j]))
            nacc = accept_greedy(torch.stack(preds), drafts)
            stats.rounds += 1
            stats.accepted.append(nacc)

            # bonus：target 在首个被拒位置的正确上下文预测 = preds[nacc]；
            # 全部草稿接受时，bonus 取 target 在末草稿位置的预测 vlogits
            if nacc < n_draft:
                bonus = torch.stack(preds)[nacc]
            else:
                bonus = greedy(vlogits[0, n_draft - 1])
            confirmed_end = pos + 1 + nacc
            new_count = nacc + 1
            room = max_new - produced
            take = min(new_count, room)

            new_tokens = list(drafts[:nacc].tolist()) + [int(bonus.item())]
            out.extend(new_tokens[:take])
            produced += take
            if produced >= max_new:
                break

            # 回滚被拒位置之后的 KV，前向 bonus 建立下轮起点
            self.cache.truncate(confirmed_end)
            blogits, bh = self.model.forward_continue(bonus.view(1), confirmed_end)
            logits_c, h_t = blogits, bh
            pos = confirmed_end
            token_next = bonus

        full = torch.cat((prompt, torch.tensor(out, device=prompt.device)))
        return full, stats
