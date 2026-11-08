"""定位修复位置后仍 lossless=False 的两个 prompt，逐轮核对（复现新草稿逻辑）。"""
import os, sys
# 项目根目录（scripts/ 的上一级）加入模块搜索路径，避免硬编码本机绝对路径。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.qwen_mtp import QwenMTPHead, target_recompute, STARTS, N_DRAFT

DEV="cuda"; DT=torch.bfloat16; GEN=64
name=os.path.join(ROOT, "models", "qwen05")
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=DT).to(DEV).eval()
head=QwenMTPHead(model).to(DT)
head.load_state_dict(torch.load(os.path.join(ROOT, "results", "qwen_mtp_head.pt")))
head.eval()

for idx in [34,37]:
    s=STARTS[idx]
    ids=tok(s,return_tensors="pt").input_ids.to(DEV)
    base=target_recompute(model,ids,GEN)[0]
    confirmed=ids[0].clone(); produced=0; bug=None
    with torch.no_grad():
        while produced<GEN and bug is None:
            full=model(input_ids=confirmed.view(1,-1),output_hidden_states=True,use_cache=False)
            C=confirmed.shape[0]
            h_t=model.model.norm(full.hidden_states[-1][0,-1]).view(1,1,-1)
            last=full.logits[0,-1]
            tok0=model.lm_head(h_t)[0,0].argmax().view(1)
            drafts,hh,nt=[tok0],h_t,tok0
            for k in range(N_DRAFT-1):
                hh,lg=head(hh,nt,position=C+k,model=model)
                nt=lg[0,0].argmax().view(1); drafts.append(nt)
            drafts=torch.cat(drafts)
            seq=torch.cat((confirmed,drafts))
            vl=model(input_ids=seq.view(1,-1),use_cache=False).logits[0]
            preds=[last.argmax()]
            for j in range(N_DRAFT-1): preds.append(vl[C+j].argmax())
            nacc=0
            while nacc<N_DRAFT and preds[nacc]==drafts[nacc]: nacc+=1
            bonus=(preds[nacc] if nacc<N_DRAFT else vl[C+N_DRAFT-1].argmax())
            # 逐位置核对
            for j in range(nacc):
                if drafts[j].item()!=base[C+j].item():
                    bug=("draft",C+j,drafts[j].item(),base[C+j].item(),"nacc",nacc); break
            if bug is None:
                bp=C+nacc
                if bp<base.shape[0] and bonus.item()!=base[bp].item():
                    bug=("bonus",bp,bonus.item(),base[bp].item(),"nacc",nacc)
            new=torch.cat((drafts[:nacc],bonus.view(1)))
            room=GEN-produced
            confirmed=torch.cat((confirmed,new[:room]))
            produced+=min(nacc+1,room)
    print(f"[{s}] {'OK' if bug is None else bug}")
