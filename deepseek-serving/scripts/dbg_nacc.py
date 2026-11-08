import os, sys
# 项目根目录（scripts/ 的上一级）加入模块搜索路径，避免硬编码本机绝对路径。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.qwen_mtp import QwenMTPHead, STARTS

DEV="cuda"; DT=torch.bfloat16; N=2
name=os.path.join(ROOT, "models", "qwen05")
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=DT).to(DEV).eval()
head=QwenMTPHead(model).to(DT)
head.load_state_dict(torch.load(os.path.join(ROOT, "results", "qwen_mtp_head.pt")))
head.eval()

ids=tok(STARTS[37],return_tensors="pt").input_ids.to(DEV)  # cat, 之前减速最多
with torch.no_grad():
    o=model(ids,use_cache=True,output_hidden_states=True)
    past=o.past_key_values
    h_t=model.model.norm(o.hidden_states[-1][0,-1]).view(1,1,-1)
    last=o.logits[0,-1]
    produced=0; rnd=0
    while produced<64:
        C=past.get_seq_length()
        tok0=model.lm_head(h_t)[0,0].argmax().view(1)
        drafts,hh,nt=[tok0],h_t,tok0
        for k in range(N-1):
            hh,lg=head(hh,nt,position=C+k,model=model)
            nt=lg[0,0].argmax().view(1); drafts.append(nt)
        drafts=torch.cat(drafts)
        v=model(input_ids=drafts.view(1,-1),past_key_values=past,use_cache=True)
        vl=v.logits[0]; vpast=v.past_key_values
        preds=[last.argmax()]
        for j in range(N-1): preds.append(vl[j].argmax())
        nacc=0
        while nacc<N and preds[nacc]==drafts[nacc]: nacc+=1
        d0ok = bool(tok0.item()==last.argmax().item())
        if nacc==0 or not d0ok:
            print(f"rnd{rnd} C={C} nacc={nacc} d0==lastpred={d0ok} tok0={tok0.item()} lastpred={last.argmax().item()} pastlen_before_verify={C}")
        remove=N-nacc
        if remove>0: vpast.crop(-remove)
        bonus=(preds[nacc] if nacc<N else vl[-1].argmax())
        o=model(input_ids=bonus.view(1,1),past_key_values=vpast,use_cache=True,output_hidden_states=True)
        past=o.past_key_values
        h_t=model.model.norm(o.hidden_states[-1][0,-1]).view(1,1,-1)
        last=o.logits[0,-1]
        new=torch.cat((drafts[:nacc],bonus.view(1)))
        room=64-produced
        produced+=min(nacc+1,room); rnd+=1
print("done rounds",rnd)
