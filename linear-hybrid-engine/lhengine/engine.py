# -*- coding: utf-8 -*-
"""
生成引擎：prefill + decode 主循环与采样。

流程：
    1. prompt 文本 -> token ids（或直接给 ids）
    2. prefill：整段前向，得到各层状态 states 与最后位置 logits
    3. 从 logits 采样出第一个新 token
    4. decode 循环：model.step 单步前向（复用固定状态），采样，追加，
       直到 eos 或达到 max_new_tokens
    5. 输出 token ids / 文本
采样支持：贪心、temperature、top-k、top-p（nucleus）。
"""
import torch
import torch.nn.functional as F

from .config import EngineConfig
from .tokenizer import load_tokenizer, encode_batch
from .cache import detach_states


def sample_logits(logits: torch.Tensor, temperature: float = 1.0,
                  top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """
    从 logits 采样单个 token。
    logits: (vocab,) 或 (b, vocab)；返回标量/一维 LongTensor。
    """
    # 贪心：温度为 0 时直接取最大
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    # top-k：只保留概率最高的 k 个
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth,
                             torch.full_like(logits, float("-inf")), logits)

    # top-p（nucleus）：按概率排序，保留累计概率刚超过 p 的最小集合
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # 累计概率超过 p 的位置移除（保留第一个越过 p 的 token）
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class Engine:
    """后 Transformer 模型的生成引擎。"""

    def __init__(self, model, cfg: EngineConfig = None, tokenizer=None,
                 device: str = None):
        self.model = model
        self.cfg = cfg or EngineConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.eval()
        self.tokenizer = tokenizer or load_tokenizer(None)

    def generate_ids(self, prompt, max_new_tokens: int = None,
                     temperature: float = None, top_k: int = None,
                     top_p: float = None, eos_id: int = None):
        """
        生成 token ids。
        prompt: 文本字符串或 token id 列表/张量
        return: 完整 token id 列表（含 prompt）
        """
        c = self.cfg
        max_new_tokens = max_new_tokens or c.max_new_tokens
        temperature = c.temperature if temperature is None else temperature
        top_k = c.top_k if top_k is None else top_k
        top_p = c.top_p if top_p is None else top_p

        # 构造输入张量
        if isinstance(prompt, str):
            input_ids = encode_batch(self.tokenizer, prompt, self.device)
        elif torch.is_tensor(prompt):
            input_ids = prompt.to(self.device).reshape(1, -1)
        else:
            input_ids = torch.tensor([prompt], device=self.device)

        generated = input_ids[0].tolist()

        # ---- prefill：整段前向 ----
        with torch.no_grad():
            logits, states = self.model(input_ids)
        states = detach_states(states)

        # 第一个新 token
        next_id = sample_logits(logits, temperature, top_k, top_p).item()
        generated.append(next_id)

        # ---- decode 循环 ----
        for _ in range(max_new_tokens - 1):
            if eos_id is not None and next_id == eos_id:
                break
            cur = torch.tensor([[next_id]], device=self.device)
            with torch.no_grad():
                logits, states = self.model.step(cur, states)
            next_id = sample_logits(logits, temperature, top_k, top_p).item()
            generated.append(next_id)

        return generated

    def generate(self, prompt: str, **kwargs) -> str:
        """生成并返回文本。"""
        ids = self.generate_ids(prompt, **kwargs)
        return self.tokenizer.decode(ids)
