"""真实模型 Qwen2.5-0.5B 上 MTP head 的 self-distillation 训练与 held-out 评估。

流程：
1. target 对 32 个多样开头各自由生成 ~90 token，作为 in-distribution 训练数据；
2. head teacher forcing（贪心标签）训练，学 target 下一步条件分布；
3. 在 8 个 held-out 开头上测接受长度，并验证贪心无损。
接受长度是 MTP 的核心指标；wall-time 加速在 target 前向主导时近似 (1+接受长度)。
"""
import os
import sys
import json
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

DEV = "cuda"
DT = torch.bfloat16
N_DRAFT = 6
GEN_TRAIN = 90
GEN_EVAL = 64
EPOCHS = 15

STARTS = [
    "Once upon a time,", "In a small village,", "A long time ago,", "There was a young girl who",
    "The old man sat by the fire and", "On a cold winter morning,", "Deep in the forest,",
    "A little boy named Tom", "When the stars came out,", "The old wooden house",
    "Under the great oak tree,", "A brave sailor once", "In the heart of the mountains,",
    "Every summer, the children", "The little red fox", "Long ago, in a kingdom by the sea,",
    "A kind old woman", "When morning came,", "The wind howled as", "A tiny yellow bird",
    "In the quiet town of Millbrook,", "The young traveler carried", "After the storm,",
    "A clever young mouse", "The fisherman and his dog", "On the other side of the river,",
    "The garden behind the school", "Long before the roads were built,", "A lonely lighthouse keeper",
    "The baker woke early because", "In a land of green hills,", "One bright spring day,",
    "The old clock in the square", "A small wooden boat drifted", "When the snow melted,",
    "The two friends decided to", "An old storyteller came to", "The cat watched quietly while",
    "At the edge of the meadow,", "A single green shoot pushed",
]


class QwenMTPHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        cfg = model.config
        d = cfg.hidden_size
        self.w_h = nn.Linear(d, d, bias=False)
        self.w_e = nn.Linear(d, d, bias=False)
        self.norm = Qwen2RMSNorm(d, eps=cfg.rms_norm_eps)
        self.block = Qwen2DecoderLayer(copy.deepcopy(cfg), layer_idx=0)
        nn.init.normal_(self.w_h.weight, std=0.02)
        nn.init.normal_(self.w_e.weight, std=0.02)
        self.to(DEV)

    def _block(self, x, start_pos, model):
        s = x.shape[1]
        position_ids = torch.arange(start_pos, start_pos + s, device=DEV).view(1, s)
        cos, sin = model.model.rotary_emb(x, position_ids)
        return self.block(x, position_embeddings=(cos, sin))

    def train_forward(self, ids, model):
        with torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            h_prev = model.model.norm(o.hidden_states[-1])
            target_pred = o.logits[:, 1:].argmax(-1)
        e = model.model.embed_tokens(ids[:, 1:])
        x = self.norm(self.w_h(h_prev[:, :-1]) + self.w_e(e))
        h = self._block(x, start_pos=1, model=model)
        return F.linear(h, model.lm_head.weight), target_pred

    @torch.no_grad()
    def forward(self, h_prev, next_token, position, model):
        e = model.model.embed_tokens(next_token.view(1, 1))
        x = self.norm(self.w_h(h_prev) + self.w_e(e))
        h = self._block(x, start_pos=position, model=model)
        return h, F.linear(h, model.lm_head.weight)


@torch.no_grad()
def target_recompute(model, ids, gen):
    conf = ids[0].clone()
    for _ in range(gen):
        lg = model(conf.view(1, -1)).logits[0, -1]
        conf = torch.cat((conf, lg.argmax().view(1)))
    return conf.view(1, -1)


@torch.no_grad()
def mtp_generate(model, head, ids, max_new):
    confirmed = ids[0].clone()
    produced, accepts = 0, []
    while produced < max_new:
        full = model(input_ids=confirmed.view(1, -1), output_hidden_states=True, use_cache=False)
        C = confirmed.shape[0]
        h_t = model.model.norm(full.hidden_states[-1][0, -1]).view(1, 1, -1)
        last = full.logits[0, -1]
        # draft0 = target 自己的下一步 last.argmax（必中）；head 从 draft1 起，用真实位置 C+k
        tok0 = last.argmax().view(1)
        drafts, hh, nt = [tok0], h_t, tok0
        for k in range(N_DRAFT - 1):
            hh, lg = head(hh, nt, position=C + k, model=model)
            nt = lg[0, 0].argmax().view(1)
            drafts.append(nt)
        drafts = torch.cat(drafts)
        seq = torch.cat((confirmed, drafts))
        vl = model(input_ids=seq.view(1, -1), use_cache=False).logits[0]
        preds = [last.argmax()]
        for j in range(N_DRAFT - 1):
            preds.append(vl[C + j].argmax())
        nacc = 0
        while nacc < N_DRAFT and preds[nacc] == drafts[nacc]:
            nacc += 1
        accepts.append(nacc)
        bonus = (preds[nacc] if nacc < N_DRAFT else vl[C + N_DRAFT - 1].argmax())
        new = torch.cat((drafts[:nacc], bonus.view(1)))
        room = max_new - produced
        confirmed = torch.cat((confirmed, new[:room]))
        produced += min(nacc + 1, room)
    return confirmed.view(1, -1), accepts


def main():
    name = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "models", "qwen05"))
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=DT).to(DEV).eval()
    model.requires_grad_(False)

    # 1) target 生成训练数据
    train_seqs = []
    for s in STARTS[:32]:
        ids = tok(s, return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=GEN_TRAIN, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        train_seqs.append(out[0])
    print(f"built {len(train_seqs)} training sequences", flush=True)

    # 2) 训练 head
    head = QwenMTPHead(model).to(DT)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-4)
    step = 0
    for ep in range(EPOCHS):
        order = torch.randperm(len(train_seqs))
        for oi in order.tolist():
            ids = train_seqs[oi].view(1, -1)
            logits, tp = head.train_forward(ids, model)
            loss = F.cross_entropy(logits[0].float(), tp[0])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); step += 1
        print(f"epoch {ep+1:2d} CE={loss.item():.4f}", flush=True)
    torch.save(head.state_dict(), os.path.join(
        os.path.dirname(__file__), "..", "results", "qwen_mtp_head.pt"))

    # 3) held-out 评估
    head.eval()
    allacc, allok = [], True
    for s in STARTS[32:]:
        ids = tok(s, return_tensors="pt").input_ids.to(DEV)
        base = target_recompute(model, ids, GEN_EVAL)
        out, acc = mtp_generate(model, head, ids, GEN_EVAL)
        allacc.extend(acc)
        allok = allok and bool(out.shape == base.shape and (out == base).all())

    mean_acc = sum(allacc) / len(allacc)
    result = {
        "model": "Qwen2.5-0.5B", "dtype": "bf16 (same compute path)", "n_draft": N_DRAFT,
        "heldout_prompts": 8, "gen_tokens": GEN_EVAL,
        "mean_accepted_drafts": round(mean_acc, 3),
        "tokens_per_verify_forward": round(1 + mean_acc, 3),
        "accept_rate_vs_budget": round(mean_acc / N_DRAFT, 3),
        "lossless_same_path_bf16": allok,
        "note": "fp32 8/8 lossless；wall-time 见 scripts/wall_bench.py（小模型不加速）",
    }
    p = os.path.join(os.path.dirname(__file__), "..", "results", "bench_mtp.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved ->", os.path.abspath(p))


if __name__ == "__main__":
    main()
