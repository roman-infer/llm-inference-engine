# -*- coding: utf-8 -*-
"""
使用真实 Mamba2 权重（或随机小模型）进行文本生成。

用法（在项目根目录）：
    # 真实权重（先用 download_weights.py 下载）
    python scripts/run_generation.py --weights weights/mamba2-130m \
        --prompt "The future of AI is" --max-new 64 --temperature 0.7

    # 离线随机小模型（无需下载，仅验证流程能跑通）
    python scripts/run_generation.py --random --prompt "hello"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lhengine.config import ModelConfig, MixerSpec, EngineConfig
from lhengine.models.lm import CausalLM
from lhengine.weights import load_safetensors_into, find_safetensors, random_init_
from lhengine.tokenizer import load_tokenizer
from lhengine.engine import Engine


def config_from_hf(weights_dir: str) -> ModelConfig:
    """从 HF 的 config.json 读取结构，构造全 mamba2 的 ModelConfig。"""
    cfg_path = os.path.join(weights_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        hf = json.load(f)

    # 不同权重版本字段名略有差异，这里做兼容取值。
    def pick(*names, default=None):
        for n in names:
            if n in hf:
                return hf[n]
        return default

    d_model = pick("d_model", default=768)
    n_layers = pick("n_layer", "num_hidden_layers", default=24)
    vocab = pick("vocab_size", default=50280)
    expand = pick("expand", default=2)
    d_state = pick("d_state", default=128)
    d_head = pick("d_head", default=64)
    conv_kernel = pick("d_conv", "conv_kernel", default=4)

    return ModelConfig(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab,
        expand=expand,
        d_state=d_state,
        d_head=d_head,
        d_key=d_head,
        conv_kernel=conv_kernel,
        mixer_pattern=MixerSpec.uniform("mamba2", n_layers),
    )


def build_random_model():
    """构造离线随机小模型（字节 tokenizer，vocab 256）。"""
    cfg = ModelConfig(
        d_model=64, n_layers=4, vocab_size=256, expand=2,
        d_head=16, d_state=16, d_key=16,
        mixer_pattern=MixerSpec.hybrid(["gdn", "gdn", "swa", "gdn"]),
    )
    model = CausalLM(cfg)
    random_init_(model, seed=1)
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="linear-hybrid-engine 生成")
    parser.add_argument("--weights", default=None, help="HF 权重目录")
    parser.add_argument("--random", action="store_true", help="用随机小模型")
    parser.add_argument("--prompt", default="Hello, world!")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.random or args.weights is None:
        print("未提供 --weights，使用离线随机小模型（输出无意义，仅验证流程）。")
        model, cfg = build_random_model()
        tokenizer = load_tokenizer(None)
    else:
        cfg = config_from_hf(args.weights)
        model = CausalLM(cfg)
        st_paths = find_safetensors(args.weights)
        if not st_paths:
            print(f"在 {args.weights} 下未找到 .safetensors 文件。")
            sys.exit(1)
        report = load_safetensors_into(model, st_paths)
        print("权重加载报告：", report)
        tokenizer = load_tokenizer(None)
        # 若目录里有 tokenizer 文件，尝试加载真实 tokenizer。
        tok_json = os.path.join(args.weights, "tokenizer.json")
        if os.path.exists(tok_json):
            tokenizer = load_tokenizer(tok_json)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(model, EngineConfig(max_new_tokens=args.max_new),
                    tokenizer=tokenizer, device=device)

    print(f"\nPrompt: {args.prompt}\n{'='*40}")
    out = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(out)


if __name__ == "__main__":
    main()
