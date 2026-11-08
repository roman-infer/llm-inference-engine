"""端到端：mini DeepSeek 真实生成 + MTP 投机无损一致性。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from node_backend.deepseek_infer.engine import Generator, LatentKVCache
from node_backend.deepseek_infer.model import DeepSeekModel, mini_config
from node_backend.deepseek_infer.mtp import MTPHead

torch.manual_seed(123)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16
CFG = mini_config()


def build(seed=123):
    torch.manual_seed(seed)
    model = DeepSeekModel(CFG, device=DEV, dtype=DT)
    head = MTPHead(CFG, model.embed_tokens, model._head(), device=DEV, dtype=DT)
    cache = LatentKVCache(CFG, device=DEV, dtype=DT)
    return Generator(model, head, cache, DEV)


def test_basic_generate():
    g = build()
    prompt = torch.tensor([1, 20, 300, 5, 90], device=DEV)
    out = g.generate(prompt, max_new=20)
    assert out.shape[0] == 25
    assert torch.isfinite(out.float()).all()
    assert (out[:5] == prompt).all()
    print(f"[OK] 普通生成 20 token: {out[5:].tolist()}")


def test_mtp_matches_target():
    # 同一模型分别用普通路径与 MTP 路径，贪心输出必须逐 token 一致（无损）
    g1 = build()
    g2 = build()
    prompt = torch.tensor([1, 20, 300, 5, 90], device=DEV)
    target_out = g1.generate(prompt, max_new=24)
    mtp_out, stats = g2.generate_mtp(prompt, max_new=24, n_draft=4)
    assert mtp_out.shape == target_out.shape, (mtp_out.shape, target_out.shape)
    mismatch = (mtp_out != target_out).nonzero()
    if len(mismatch):
        i = mismatch[0].item()
        raise AssertionError(
            f"MTP 与 target 在位置 {i} 不一致: mtp={mtp_out[i].item()} "
            f"target={target_out[i].item()}")
    print(f"[OK] MTP 投机输出与 target 完全一致（贪心无损），"
          f"各轮接受数={stats.accepted} 平均={stats.mean_accept:.2f}")


def test_chunk_vs_decode():
    """forward_continue（chunk）与逐 token decode 的同位置结果应一致。"""
    model = DeepSeekModel(CFG, device=DEV, dtype=DT)
    cache = LatentKVCache(CFG, device=DEV, dtype=DT)
    model.cache = cache
    prompt = torch.tensor([1, 20, 300, 5, 90, 7, 8], device=DEV)
    model.forward_prefill(prompt)
    P = prompt.shape[0]

    chunk = torch.tensor([11, 22, 33], device=DEV)
    cl, _ = model.forward_continue(chunk, P)

    # decode 路径：重置后逐 token
    cache.reset()
    model.forward_prefill(prompt)
    outs = []
    logits = None
    for i, t in enumerate(chunk.tolist()):
        tt = torch.tensor(t, device=DEV)
        dl, _ = model.forward_decode(tt, P + i)
        outs.append(dl[0, 0])
    dl_all = torch.stack(outs)
    err = ((cl[0].float() - dl_all.float()).norm() / dl_all.float().norm()).item()
    assert err < 0.05, f"chunk 与 decode 不一致: {err}"
    print(f"[OK] chunk 前向与逐 token decode 一致 (err={err:.4f})")


if __name__ == "__main__":
    test_basic_generate()
    test_mtp_matches_target()
    test_chunk_vs_decode()
    print("\n全部端到端测试通过 ✅")
