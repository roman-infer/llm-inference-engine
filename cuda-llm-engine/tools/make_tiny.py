#!/usr/bin/env python3
"""生成一个微型*随机权重*模型 + 最小分词器，无需联网，
以便端到端验证完整流水线（checkpoint -> model -> attention -> sampling ->
text）。输出内容按设计就是乱码；仅用于验证形状和数值正确性。

用法：
    python make_tiny.py --arch gpt2  --out ../models/tiny-gpt2
    python make_tiny.py --arch llama --out ../models/tiny-llama
"""

import argparse
import json
import os
import struct

import numpy as np


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def write_tokenizer(out):
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    vocab["<|endoftext|>"] = 256
    with open(os.path.join(out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(out, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")


def save_safetensors(tensors, path):
    """tensors：dict 名称 -> (np.ndarray, 数据类型字符串)。"""
    header = {}
    blob = bytearray()
    offset = 0
    for name, (arr, dtype) in tensors.items():
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        blob += arr.tobytes()
        offset += nbytes
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)
    print(f"Wrote {path} ({offset/1e3:.1f} KB, {len(header)} tensors)")


def randn(*shape, scale=0.1):
    return (np.random.randn(*shape) * scale).astype(np.float16)


def build_gpt2():
    L, H, E, I, V, S = 2, 4, 64, 128, 257, 64
    t = {
        "wte": (randn(V, E, scale=0.05), "F16"),
        "wpe": (randn(S, E), "F16"),
    }
    for i in range(L):
        p = f"h{i}."
        t[p+"ln1.w"] = (np.ones(E, np.float16), "F16")
        t[p+"ln1.b"] = (np.zeros(E, np.float16), "F16")
        t[p+"attn.qkv.w"] = (randn(3*E, E), "F16")
        t[p+"attn.qkv.b"] = (np.zeros(3*E, np.float16), "F16")
        t[p+"attn.proj.w"] = (randn(E, E), "F16")
        t[p+"attn.proj.b"] = (np.zeros(E, np.float16), "F16")
        t[p+"ln2.w"] = (np.ones(E, np.float16), "F16")
        t[p+"ln2.b"] = (np.zeros(E, np.float16), "F16")
        t[p+"mlp.fc.w"] = (randn(I, E), "F16")
        t[p+"mlp.fc.b"] = (np.zeros(I, np.float16), "F16")
        t[p+"mlp.proj.w"] = (randn(E, I), "F16")
        t[p+"mlp.proj.b"] = (np.zeros(E, np.float16), "F16")
    t["lnf.w"] = (np.ones(E, np.float16), "F16")
    t["lnf.b"] = (np.zeros(E, np.float16), "F16")
    cfg = dict(n_layer=L, n_head=H, n_kv_head=H, n_embd=E, n_inter=I,
               max_seq_len=S, vocab_size=V, eps=1e-5, theta=10000.0)
    return t, cfg


def build_llama():
    L, H, KH, E, I, V, S = 2, 4, 2, 64, 128, 257, 64
    d = E // H
    t = {"tok_emb": (randn(V, E, scale=0.05), "F16")}
    for i in range(L):
        p = f"layers{i}."
        t[p+"attn_norm.w"] = (np.ones(E, np.float16), "F16")
        t[p+"q.w"] = (randn(H*d, E), "F16")
        t[p+"k.w"] = (randn(KH*d, E), "F16")
        t[p+"v.w"] = (randn(KH*d, E), "F16")
        t[p+"o.w"] = (randn(E, H*d), "F16")
        t[p+"mlp_norm.w"] = (np.ones(E, np.float16), "F16")
        t[p+"gate.w"] = (randn(I, E), "F16")
        t[p+"up.w"] = (randn(I, E), "F16")
        t[p+"down.w"] = (randn(E, I), "F16")
    t["norm.w"] = (np.ones(E, np.float16), "F16")
    t["lm_head"] = (randn(V, E, scale=0.05), "F16")
    cfg = dict(n_layer=L, n_head=H, n_kv_head=KH, n_embd=E, n_inter=I,
               max_seq_len=S, vocab_size=V, eps=1e-5, theta=10000.0)
    return t, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["gpt2", "llama"], default="gpt2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tensors, cfg = build_gpt2() if args.arch == "gpt2" else build_llama()
    save_safetensors(tensors, os.path.join(args.out, "model.safetensors"))
    write_tokenizer(args.out)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("Tiny", args.arch, "model + tokenizer written to", args.out)


if __name__ == "__main__":
    main()
