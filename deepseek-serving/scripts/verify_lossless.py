"""验证 MTP 贪心无损：baseline 与 MTP 走同一整序列计算路径（消除 bf16 路径差异）。"""
import os, sys
# 项目根目录（scripts/ 的上一级）加入模块搜索路径，避免硬编码本机绝对路径。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.qwen_mtp import QwenMTPHead, mtp_generate, TEXTS

DEV="cuda"; DT=torch.bfloat16
name=os.path.join(ROOT, "models", "qwen05")
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=DT).to(DEV).eval()
head=QwenMTPHead(model).to(DT)
head.load_state_dict(torch.load(os.path.join(ROOT, "results", "qwen_mtp_head.pt")))
head.eval()

@torch.no_grad()
def target_recompute(ids, gen):
    """整序列重算的贪心 rollout（与 MTP 同计算路径）。"""
    conf=ids[0].clone()
    for _ in range(gen):
        lg=model(conf.view(1,-1)).logits[0,-1]
        conf=torch.cat((conf,lg.argmax().view(1)))
    return conf.view(1,-1)

gen=40; allok=True; allacc=[]
for ti,pr in enumerate([t.split(",")[0]+"," for t in TEXTS[:6]]):
    ids=tok(pr,return_tensors="pt").input_ids.to(DEV)
    base=target_recompute(ids,gen)
    out,acc=mtp_generate(model,head,ids,gen)
    allacc.extend(acc)
    ok=out.shape==base.shape and (out==base).all()
    allok=allok and ok
    print(f"prompt{ti}: lossless={ok} accepts={acc}")
print("\nmean accepted/round:", round(sum(allacc)/len(allacc),3))
print("=== ALL LOSSLESS (same compute path) ===" if allok else "=== MISMATCH ===")
