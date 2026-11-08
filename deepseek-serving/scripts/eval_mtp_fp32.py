"""fp32 下验证 MTP 贪心无损（消除 bf16 形状相关的累加舍入差异）。"""
import os, sys, json
# 项目根目录（scripts/ 的上一级）加入模块搜索路径，避免硬编码本机绝对路径。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.qwen_mtp import QwenMTPHead, target_recompute, mtp_generate, STARTS, N_DRAFT

DEV="cuda"; GEN=48
name=os.path.join(ROOT, "models", "qwen05")
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(DEV).eval()
head=QwenMTPHead(model)   # fp32
head.load_state_dict(torch.load(os.path.join(ROOT, "results", "qwen_mtp_head.pt")))
head=head.to(torch.float32).eval()

allacc=[]; allok=True
for s in STARTS[32:]:
    ids=tok(s,return_tensors="pt").input_ids.to(DEV)
    base=target_recompute(model,ids,GEN)
    out,acc=mtp_generate(model,head,ids,GEN)
    allacc.extend(acc)
    ok=bool(out.shape==base.shape and (out==base).all())
    allok=allok and ok
    print(f"{s:35s} lossless={ok} meanacc={sum(acc)/len(acc):.2f}",flush=True)
ma=sum(allacc)/len(allacc)
print(json.dumps({"dtype":"fp32","mean_accepted_per_round":round(ma,3),
                  "mean_accepted_length_draft":round(ma,3),
                  "tokens_per_verify_forward":round(ma+1,3),"lossless":allok},indent=2))
