#!/usr/bin/env python3
"""将 HuggingFace GPT-2 模型导出为本引擎使用的 safetensors 布局，FP16 精度。
同时将 vocab.json / merges.txt 拷贝到权重文件旁边。

用法：
    python convert_gpt2.py --model gpt2 --out ../models/gpt2
"""

import argparse
import json
import os
import shutil
import struct

import torch


_DTYPE = {
    torch.float16: "F16",
    torch.float32: "F32",
}


def save_safetensors(tensors, path):
    """tensors：dict 名称 -> torch.Tensor（连续存储）。"""
    header = {}
    blob = bytearray()
    offset = 0
    for name, tensor in tensors.items():
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": _DTYPE[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        blob += tensor.cpu().numpy().tobytes()
        offset += nbytes

    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(blob)
    print(f"Wrote {path} ({offset/1e6:.1f} MB, {len(header)} tensors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--out", default="../models/gpt2")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = AutoModelForCausalLM.from_pretrained(args.model)
    hf = hf.half().eval()
    m = hf.transformer if hasattr(hf, "transformer") else hf.model

    tensors = {
        "wte": m.wte.weight.data,
        "wpe": m.wpe.weight.data,
    }

    for i, block in enumerate(m.h):
        p = f"h{i}."
        tensors[p + "ln1.w"] = block.ln_1.weight.data
        tensors[p + "ln1.b"] = block.ln_1.bias.data

        qkv_w = block.attn.c_attn.weight.data
        qkv_b = block.attn.c_attn.bias.data
        tensors[p + "attn.qkv.w"] = qkv_w
        tensors[p + "attn.qkv.b"] = qkv_b
        tensors[p + "attn.proj.w"] = block.attn.c_proj.weight.data
        tensors[p + "attn.proj.b"] = block.attn.c_proj.bias.data

        tensors[p + "ln2.w"] = block.ln_2.weight.data
        tensors[p + "ln2.b"] = block.ln_2.bias.data
        tensors[p + "mlp.fc.w"] = block.mlp.c_fc.weight.data
        tensors[p + "mlp.fc.b"] = block.mlp.c_fc.bias.data
        tensors[p + "mlp.proj.w"] = block.mlp.c_proj.weight.data
        tensors[p + "mlp.proj.b"] = block.mlp.c_proj.bias.data

    tensors["lnf.w"] = m.ln_f.weight.data
    tensors["lnf.b"] = m.ln_f.bias.data

    save_safetensors(tensors, os.path.join(args.out, "model.safetensors"))

    tok = AutoTokenizer.from_pretrained(args.model)
    for fname in ("vocab.json", "merges.txt"):
        src = tok.get_vocab_files(None) if False else None
    # 分词器文件位于 HF 缓存目录中；save_vocabulary 会自动将其拷贝出来。
    files = tok.save_vocabulary(args.out)
    for f in files:
        if f:
            shutil.copy(f, os.path.join(args.out, os.path.basename(f)))
    print("Tokenizer files written to", args.out)


if __name__ == "__main__":
    main()
