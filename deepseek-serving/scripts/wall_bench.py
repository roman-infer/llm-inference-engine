"""扫描草稿长度 N，找 MTP wall-time 最优点（草稿预算应匹配接受长度）。"""
import os, sys, json, time
# 项目根目录（scripts/ 的上一级）加入模块搜索路径，避免硬编码本机绝对路径。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.qwen_mtp import QwenMTPHead, STARTS

DEV="cuda"; DT=torch.bfloat16; GEN=64
name=os.path.join(ROOT, "models", "qwen05")
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=DT).to(DEV).eval()
head=QwenMTPHead(model).to(DT)
head.load_state_dict(torch.load(os.path.join(ROOT, "results", "qwen_mtp_head.pt")))
head.eval()


@torch.no_grad()
def mtp_fast(ids, max_new, N):
    o=model(ids,use_cache=True,output_hidden_states=True)
    past=o.past_key_values
    h_t=model.model.norm(o.hidden_states[-1][0,-1]).view(1,1,-1)
    last=o.logits[0,-1]
    out=ids[0].clone(); produced=0; accepts=[]
    while produced<max_new:
        C=past.get_seq_length()
        tok0=last.argmax().view(1)
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
        accepts.append(nacc)
        remove=N-nacc
        if remove>0: vpast.crop(-remove)
        bonus=(preds[nacc] if nacc<N else vl[-1].argmax())
        o=model(input_ids=bonus.view(1,1),past_key_values=vpast,use_cache=True,
                output_hidden_states=True)
        past=o.past_key_values
        h_t=model.model.norm(o.hidden_states[-1][0,-1]).view(1,1,-1)
        last=o.logits[0,-1]
        new=torch.cat((drafts[:nacc],bonus.view(1)))
        room=max_new-produced
        out=torch.cat((out,new[:room])); produced+=min(nacc+1,room)
    return out.view(1,-1),accepts


# baseline 时间（每个 prompt 一次，所有 N 共用）
prompts=[tok(s,return_tensors="pt").input_ids.to(DEV) for s in STARTS[32:]]
tg=[]; bases=[]
for ids in prompts:
    torch.cuda.synchronize(); t0=time.perf_counter()
    b=model.generate(ids,max_new_tokens=GEN,do_sample=False,pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); tg.append(time.perf_counter()-t0); bases.append(b)

summary=[]
for N in [2,3,4,6]:
    mt=[]; allacc=[]; nmatch=0
    for i,ids in enumerate(prompts):
        torch.cuda.synchronize(); t0=time.perf_counter()
        out,acc=mtp_fast(ids,GEN,N)
        torch.cuda.synchronize(); mt.append(time.perf_counter()-t0)
        allacc.extend(acc)
        if out.shape==bases[i].shape and (out==bases[i]).all(): nmatch+=1
    ma=sum(allacc)/len(allacc)
    r={"N":N,"mean_accepted_drafts":round(ma,3),
       "tokens_per_round":round(ma+1,3),
       "walltime_speedup":round(sum(tg)/sum(mt),3),
       "lossless_prompts":f"{nmatch}/8"}
    summary.append(r)
    print(r,flush=True)
p=os.path.join(ROOT, "results", "bench_mtp_sweep.json")
with open(p,"w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
print("saved")
