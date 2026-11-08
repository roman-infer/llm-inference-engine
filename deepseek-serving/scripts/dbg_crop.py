import os
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
DEV="cuda"; DT=torch.bfloat16
# 项目根目录（scripts/ 的上一级），模型路径相对项目根定位，避免硬编码本机绝对路径。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = os.path.join(ROOT, "models", "qwen05")
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=DT).to(DEV).eval()
ids=tok("Once upon a time there was a little rabbit who lived", return_tensors="pt").input_ids.to(DEV)
cache=DynamicCache()
model(input_ids=ids, past_key_values=cache, use_cache=True)
print("len before crop:", cache.get_seq_length())
cache.crop(3)
print("len after crop(3):", cache.get_seq_length(), "(expect", ids.shape[1]-3, ")")
# 检查底层 layer 存储形状
l0 = cache.layers[0] if hasattr(cache,"layers") else None
print("layer0 attrs:", [a for a in dir(l0) if not a.startswith("__")][:15] if l0 else "no layers")
