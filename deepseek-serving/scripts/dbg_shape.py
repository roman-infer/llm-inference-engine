import os, sys, copy
# 项目根目录（scripts/ 的上一级）加入模块搜索路径，避免硬编码本机绝对路径。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

DEV="cuda"; DT=torch.bfloat16
name=os.path.join(ROOT, "models", "qwen05")
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=DT).to(DEV).eval()
block=Qwen2DecoderLayer(copy.deepcopy(model.config),0).to(DEV).to(DT)

ids=tok("Once upon a time there was a little rabbit", return_tensors="pt").input_ids.to(DEV)
s=ids.shape[1]
x=torch.randn(1,s,896,device=DEV,dtype=DT)
position_ids=torch.arange(1,1+s,device=DEV).view(1,s)
cos,sin=model.model.rotary_emb(x,position_ids)
print("x",x.shape,"cos",cos.shape,"sin",sin.shape)
out=block(x, position_embeddings=(cos,sin))[0]
print("block out",out.shape)
