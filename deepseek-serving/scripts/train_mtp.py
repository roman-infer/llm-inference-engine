"""训练 MTP head 对齐 target（teacher forcing，贪心标签）。

target 冻结；在随机序列上让 head 学会 target 的下一步 token 预测。
接受率反映 head 与 target 的对齐程度，与 target 是否具语义无关；
训练后的接受率/加速比是 MTP 机制在"head 已对齐"下的真实能力数据。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from node_backend.deepseek_infer.model import DeepSeekModel, mini_config
from node_backend.deepseek_infer.mtp import MTPHead

DEV = "cuda"
DT = torch.float32
SEQ = 64
N_POS = 8
STEPS = 600
LR = 5e-4


def main():
    torch.manual_seed(123)
    cfg = mini_config()
    target = DeepSeekModel(cfg, device=DEV, dtype=DT)
    target.requires_grad_(False)
    head = MTPHead(cfg, target.embed_tokens, target._head(), device=DEV, dtype=DT)
    head.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)

    for step in range(STEPS):
        tokens = torch.randint(0, cfg.vocab_size, (SEQ,), device=DEV)
        with torch.no_grad():
            logits, hiddens = target.forward_prefill(tokens)
            labels = logits[0].argmax(-1)          # target 各位置贪心预测
        positions = torch.randint(0, SEQ - 2, (N_POS,))
        loss = 0.0
        for t in positions.tolist():
            _, l = head(hiddens[0, t], tokens[t + 1], moe_mode="naive")
            loss = loss + F.cross_entropy(l[0, 0].float(), labels[t + 1])
        loss = loss / N_POS
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            print(f"step {step:4d}  CE={loss.item():.4f}", flush=True)

    out = os.path.join(os.path.dirname(__file__), "..", "results", "mtp_head.pt")
    torch.save(head.state_dict(), out)
    print(f"saved -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
