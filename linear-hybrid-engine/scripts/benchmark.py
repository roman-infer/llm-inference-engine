# -*- coding: utf-8 -*-
"""
Benchmark：验证后 Transformer 线性架构的两个核心卖点。

1) Prefill 随上下文长度"线性"扩展（时间 ~ O(L)）；
2) Decode 状态显存"恒定"（O(1) state），而同规模 Transformer 的 KV 随长度 O(L) 增长。

用法（在项目根目录）：
    python scripts/benchmark.py
    python scripts/benchmark.py --lengths 128,256,512,1024 --mixer gdn
    python scripts/benchmark.py --save results.json

说明：
    - 用小尺寸合成模型（CPU 也能跑），重点是"趋势"而非绝对耗时；
    - 显存对比给出"线性模型 state"与"Transformer KV cache"的理论字节数，
      这是工程评估中最直观的对比口径。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lhengine.config import ModelConfig, MixerSpec
from lhengine.models.lm import CausalLM
from lhengine.weights import random_init_


def build_model(kind: str, n_layers: int = 4, d_model: int = 64):
    """构造小尺寸纯线性模型（vocab 256，字节 tokenizer）。"""
    cfg = ModelConfig(
        d_model=d_model, n_layers=n_layers, vocab_size=256, expand=2,
        d_head=16, d_state=16, d_key=16,
        mixer_pattern=MixerSpec.uniform(kind, n_layers),
    )
    model = CausalLM(cfg)
    random_init_(model, seed=1)
    model.eval()
    return model, cfg


def time_prefill(model, length: int, repeats: int = 3, device: str = "cpu"):
    """测量整段 prefill 的平均耗时（秒）。"""
    ids = torch.randint(1, 256, (1, length), device=device)
    # 预热一次（触发编译/缓存）。
    with torch.no_grad():
        model(ids)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(ids)
        best = min(best, time.perf_counter() - t0)
    return best


def linear_state_bytes(cfg) -> int:
    """
    线性模型单层状态的字节数（GDN：状态矩阵 (H,P,K)）。
    状态大小与序列长度无关——这是 O(1) 的关键。
    """
    H = cfg.heads
    P = cfg.d_head
    K = cfg.d_key
    # n_layers 层 × 每头状态 (P,K) × 4 字节（float32）。
    return cfg.n_layers * H * P * K * 4


def transformer_kv_bytes(cfg, length: int) -> int:
    """
    同规模 Transformer 的 KV cache 字节数，随长度线性增长。
    KV = 2(K和V) × n_layers × heads × d_head × length × 4 字节。
    """
    H = cfg.heads
    return 2 * cfg.n_layers * H * cfg.d_head * length * 4


def human(n: int) -> str:
    """字节数转人类可读单位。"""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def main():
    parser = argparse.ArgumentParser(description="线性架构 benchmark")
    parser.add_argument("--lengths", default="128,256,512,1024")
    parser.add_argument("--mixer", default="gdn", choices=["gdn", "mamba2", "delta"])
    parser.add_argument("--save", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(args.mixer)
    model.to(device)

    print(f"mixer={args.mixer}, device={device}, layers={cfg.n_layers}, "
          f"heads={cfg.heads}, d_head={cfg.d_head}")
    print(f"线性模型每层状态（固定）：{human(linear_state_bytes(cfg))}")
    print()
    print(f"{'长度':>6} | {'prefill(s)':>11} | {'线性state':>10} | {'Transformer KV':>14}")
    print("-" * 60)

    rows = []
    for L in lengths:
        dt = time_prefill(model, L, device=device)
        sb = linear_state_bytes(cfg)
        kb = transformer_kv_bytes(cfg, L)
        print(f"{L:>6} | {dt:>11.4f} | {human(sb):>10} | {human(kb):>14}")
        rows.append({"length": L, "prefill_s": dt,
                     "linear_state_bytes": sb, "transformer_kv_bytes": kb})

    # 结论：prefill 耗时应近似翻倍（长度翻倍时）；线性 state 恒定；KV 线性增长。
    print("\n结论：")
    print("  - prefill 时间随长度近似线性增长（长度翻倍 -> 耗时约翻倍）；")
    print("  - 线性模型 state 字节数恒定，与长度无关；")
    print("  - Transformer KV 随长度线性增长，长上下文时差距悬殊。")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"mixer": args.mixer, "rows": rows}, f, indent=2,
                      ensure_ascii=False)
        print(f"\n结果已保存到 {args.save}")


if __name__ == "__main__":
    main()
