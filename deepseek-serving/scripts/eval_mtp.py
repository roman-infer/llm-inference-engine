"""加载训练好的 head，在 8 个 held-out 开头上评估接受长度与无损性。"""
import os, sys, json
# 项目根目录（scripts/ 的上一级）加入模块搜索路径，避免硬编码本机绝对路径。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.qwen_mtp import QwenMTPHead, target_recompute, mtp_generate, STARTS, N_DRAFT

DEV="cuda"; DT=torch.bfloat16; GEN_EVAL=64
name=os.path.join(ROOT, "models", "qwen05")
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=DT).to(DEV).eval()
head=QwenMTPHead(model).to(DT)
head.load_state_dict(torch.load(os.path.join(ROOT, "results", "qwen_mtp_head.pt")))
head.eval()

allacc=[]; allok=True
for s in STARTS[32:]:
    ids=tok(s,return_tensors="pt").input_ids.to(DEV)
    base=target_recompute(model,ids,GEN_EVAL)
    out,acc=mtp_generate(model,head,ids,GEN_EVAL)
    allacc.extend(acc)
    ok=bool(out.shape==base.shape and (out==base).all())
    allok=allok and ok
    print(f"{s:35s} lossless={ok} accepts={acc}", flush=True)

mean_acc=sum(allacc)/len(allacc)
result={
    "model":"Qwen2.5-0.5B","n_draft":N_DRAFT,"heldout_prompts":8,"gen_tokens":GEN_EVAL,
    "mean_accepted_per_round":round(mean_acc,3),
    "mean_accepted_length":round(1+mean_acc,3),
    "accept_rate_vs_budget":round(mean_acc/N_DRAFT,3),
    "theoretical_target_forward_reduction":round(1/(1+mean_acc),3),
    "lossless_greedy_match":allok,
}
p=os.path.join(ROOT, "results", "bench_mtp.json")
with open(p,"w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
