"""Kernel benchmark：FP8 / MLA / MoE / 端到端，全部真实计时落 JSON。"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from node_backend.deepseek_infer.utils import benchmark_cuda
from node_backend.deepseek_infer.fp8 import FP8Linear
from node_backend.deepseek_infer.mla import MLA, MLAConfig
from node_backend.deepseek_infer.moe import MoE, MoEConfig
from node_backend.deepseek_infer.model import DeepSeekModel, mini_config
from node_backend.deepseek_infer.engine import LatentKVCache

DEV = "cuda"
DT = torch.float16
RESULTS = {}


def frob(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# ---------------- FP8 ----------------
def bench_fp8():
    sizes = [(128, 2048, 2048), (512, 4096, 4096), (1024, 4096, 8192)]
    out = []
    for M, K, N in sizes:
        lin = FP8Linear(K, N, block_size=128, bias=False, device=DEV, dtype=DT)
        x = torch.randn(M, K, device=DEV, dtype=DT) * 0.5
        ref = lin(x, "fp16")
        r_t = benchmark_cuda(lambda: lin(x, "fp16"), name="fp16")
        r_s = benchmark_cuda(lambda: lin(x, "tensorwise"), name="tensorwise")
        r_b = benchmark_cuda(lambda: lin(x, "blockwise"), name="blockwise")
        flops = 2 * M * N * K
        out.append({
            "size": f"M{M}xK{K}xN{N}",
            "fp16_ms": round(r_t.median_ms, 4),
            "tensorwise_ms": round(r_s.median_ms, 4),
            "blockwise_ms": round(r_b.median_ms, 4),
            "fp16_TFLOPS": round(flops / r_t.median_ms / 1e9, 2),
            "tensorwise_TFLOPS": round(flops / r_s.median_ms / 1e9, 2),
            "tensorwise_cos": round(cos(lin(x, "tensorwise"), ref), 4),
            "blockwise_cos": round(cos(lin(x, "blockwise"), ref), 4),
            "tensorwise_frob": round(frob(lin(x, "tensorwise"), ref), 4),
            "blockwise_frob": round(frob(lin(x, "blockwise"), ref), 4),
        })
        print(f"[FP8] M{M} K{K} N{N}: fp16={r_t.median_ms:.3f}ms "
              f"tensor={r_s.median_ms:.3f}ms block={r_b.median_ms:.3f}ms "
              f"cos={out[-1]['tensorwise_cos']}")
    RESULTS["fp8"] = out


# ---------------- MLA ----------------
def bench_mla():
    cfg = mini_config()
    mla = MLA(MLAConfig(
        d=cfg.d_model, n_h=cfg.n_heads, d_h=cfg.d_head,
        d_hR=cfg.d_head_rope, d_c=cfg.d_latent,
        d_cq=cfg.d_latent_q),
        max_seq_len=cfg.max_seq_len, device=DEV, dtype=DT)

    # 传统 MHA 每 token KV 字节 vs MLA latent
    head_total = cfg.d_head + cfg.d_head_rope
    trad = 2 * cfg.n_heads * head_total * 2
    latent = (cfg.d_latent + cfg.n_heads * cfg.d_head_rope) * 2
    out = {"kv_compression": round(trad / latent, 2),
           "traditional_bytes_per_token": trad,
           "latent_bytes_per_token": latent, "scans": []}

    for T in (64, 128, 256, 512, 1024):
        h = torch.randn(1, T, cfg.d_model, device=DEV, dtype=DT)
        ckv_all = torch.nn.functional.linear(h[0], mla.w_dkv)          # (T,d_c)
        krope_pre = torch.nn.functional.linear(h[0], mla.w_kr).view(
            T, cfg.n_heads, cfg.d_head_rope)
        from node_backend.deepseek_infer.mla.rope import apply_rope
        krope_all = apply_rope(krope_pre, mla.rope_cos, mla.rope_sin,
                               positions=torch.arange(T, device=DEV))
        ckv_hist = ckv_all[:-1]; krope_hist = krope_all[:-1]
        h_t = h[0, T - 1]
        rn = benchmark_cuda(lambda: mla.decode_naive(h_t, ckv_hist, krope_hist, T - 1),
                            warmup=5, iters=20)
        ra = benchmark_cuda(lambda: mla.decode_absorbed(h_t, ckv_hist, krope_hist, T - 1),
                            warmup=5, iters=20)
        on = mla.decode_naive(h_t, ckv_hist, krope_hist, T - 1)
        oa = mla.decode_absorbed(h_t, ckv_hist, krope_hist, T - 1)
        out["scans"].append({
            "history": T - 1,
            "naive_ms": round(rn.median_ms, 4),
            "absorbed_ms": round(ra.median_ms, 4),
            "absorbed_vs_naive_frob": round(frob(oa, on), 5),
        })
        print(f"[MLA] T={T-1}: naive={rn.median_ms:.3f}ms absorbed={ra.median_ms:.3f}ms")
    RESULTS["mla"] = out


# ---------------- MoE ----------------
def bench_moe():
    cfg = mini_config()
    moe = MoE(MoEConfig(cfg.d_model, cfg.moe_inter, cfg.n_routed,
                        cfg.n_active, cfg.n_shared), device=DEV, dtype=DT)
    out = []
    for N in (32, 64, 128, 256):
        x = torch.randn(N, cfg.d_model, device=DEV, dtype=DT)
        rg = benchmark_cuda(lambda: moe(x, "grouped"), warmup=5, iters=30)
        rn = benchmark_cuda(lambda: moe(x, "naive"), warmup=3, iters=10)
        out.append({"tokens": N, "grouped_ms": round(rg.median_ms, 4),
                    "naive_ms": round(rn.median_ms, 4),
                    "speedup": round(rn.median_ms / rg.median_ms, 2),
                    "frob": round(frob(moe(x, "grouped"), moe(x, "naive")), 5)})
        print(f"[MoE] N={N}: grouped={rg.median_ms:.3f}ms naive={rn.median_ms:.3f}ms "
              f"speedup={out[-1]['speedup']}x")
    RESULTS["moe"] = out


# ---------------- 端到端 ----------------
def bench_e2e():
    torch.manual_seed(123)
    cfg = mini_config()
    model = DeepSeekModel(cfg, device=DEV, dtype=DT)
    cache = LatentKVCache(cfg, device=DEV, dtype=DT)
    model.cache = cache
    out = {}
    for S in (256, 512):
        toks = torch.randint(0, cfg.vocab_size, (S,), device=DEV)
        cache.reset()
        r = benchmark_cuda(lambda: model.forward_prefill(toks), warmup=3, iters=10)
        out[f"prefill_{S}"] = {"ms": round(r.median_ms, 3),
                               "tok_per_s": round(S / r.median_ms * 1000, 1)}
    # decode 单 token（历史 512）
    toks = torch.randint(0, cfg.vocab_size, (512,), device=DEV)
    cache.reset()
    model.forward_prefill(toks)
    ht = torch.randn(cfg.d_model, device=DEV, dtype=DT)
    r = benchmark_cuda(lambda: model.forward_decode(
        torch.tensor(7, device=DEV), 512), warmup=5, iters=50)
    out["decode_single"] = {"ms": round(r.median_ms, 3),
                            "tok_per_s": round(1000 / r.median_ms, 1)}
    print(f"[E2E] {out}")
    RESULTS["e2e"] = out


if __name__ == "__main__":
    bench_fp8()
    bench_mla()
    bench_moe()
    bench_e2e()
    p = os.path.join(os.path.dirname(__file__), "..", "results", "bench_kernels.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {os.path.abspath(p)}")
