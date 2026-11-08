#!/usr/bin/env python3
"""将 HuggingFace LLaMA 风格模型导出为本引擎使用的 safetensors 布局，FP16。

    python convert_llama.py --model meta-llama/Llama-3.2-1B --out ../models/llama

关于分词器的说明：上游 LLaMA 使用 SentencePiece（.model 格式），本引擎
自带的 BPE 加载器无法解析。流水线测试请使用
tools/make_tiny.py --arch llama（生成兼容 GPT-BPE 的最小词表）；
如需真实 LLaMA 词表，需将 SentencePiece 转换为 vocab.json/merges.txt
（例如通过 HuggingFace 慢速分词器导出），并将这两个文件放到输出目录。
"""

import argparse
import json
import os
import struct


_DTYPE = {
    "float16": "F16",
    "float32": "F32",
}


def save_safetensors(tensors, path):
    header = {}
    blob = bytearray()
    offset = 0
    for name, tensor in tensors.items():
        tensor = tensor.contiguous().cpu().numpy()
        nbytes = tensor.nbytes
        header[name] = {
            "dtype": "F16",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        blob += tensor.tobytes()
        offset += nbytes
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)
    print(f"Wrote {path} ({offset/1e6:.1f} MB, {len(header)} tensors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).eval()
    base = hf.model if hasattr(hf, "model") else hf.transformer

    tensors = {"tok_emb": base.embed_tokens.weight.data}
    for i, layer in enumerate(base.layers):
        p = f"layers{i}."
        tensors[p+"attn_norm.w"] = layer.input_layernorm.weight.data
        tensors[p+"q.w"] = layer.self_attn.q_proj.weight.data
        tensors[p+"k.w"] = layer.self_attn.k_proj.weight.data
        tensors[p+"v.w"] = layer.self_attn.v_proj.weight.data
        tensors[p+"o.w"] = layer.self_attn.o_proj.weight.data
        tensors[p+"mlp_norm.w"] = layer.post_attention_layernorm.weight.data
        tensors[p+"gate.w"] = layer.mlp.gate_proj.weight.data
        tensors[p+"up.w"] = layer.mlp.up_proj.weight.data
        tensors[p+"down.w"] = layer.mlp.down_proj.weight.data
    tensors["norm.w"] = base.norm.weight.data
    if getattr(hf, "lm_head", None) is not None:
        tensors["lm_head"] = hf.lm_head.weight.data

    save_safetensors(tensors,
                     os.path.join(args.out, "model.safetensors"))


if __name__ == "__main__":
    main()
