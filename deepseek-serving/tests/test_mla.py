"""MLA 数值正确性：RoPE、prefill、naive/absorbed decode 三路径一致性、压缩比。"""
import torch

from common import frobenius_rel_err
from node_backend.deepseek_infer.mla import MLA, MLAConfig
from node_backend.deepseek_infer.mla.rope import apply_rope, precompute_rope

torch.manual_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16

CFG = MLAConfig(d=64, n_h=4, d_h=16, d_hR=8, d_c=32, d_cq=32)


def test_rope():
    dim = 4
    cos, sin = precompute_rope(dim, 8, device=DEV)
    # x = [1,1,0,0]（GPT-J: x1=[1,1], x2=[0,0]）-> [cos,cos,sin,sin]
    x = torch.tensor([[1.0, 1.0, 0.0, 0.0]], device=DEV).view(1, 1, 1, dim)
    r = apply_rope(x, cos, sin, positions=torch.tensor([1])).squeeze()
    import math
    a = math.cos(1.0)
    b = math.sin(1.0)
    assert abs(r[0].item() - a) < 1e-2 and abs(r[2].item() - b) < 1e-2, r
    # 模长不变
    x2 = torch.randn(1, 5, 2, dim, device=DEV)
    r2 = apply_rope(x2, cos, sin)
    assert torch.allclose(x2.norm(dim=-1), r2.norm(dim=-1), atol=1e-2)
    print("[OK] RoPE 旋转角度与模长保持性质正确")


def test_prefill():
    mla = MLA(CFG, max_seq_len=64, device=DEV, dtype=DT)
    h = torch.randn(1, 12, CFG.d, device=DEV, dtype=DT)
    out, ckv, krope = mla.forward_prefill(h)
    assert out.shape == (1, 12, CFG.d)
    assert ckv.shape == (1, 12, CFG.d_c)
    assert krope.shape == (1, 12, CFG.n_h, CFG.d_hR)
    assert torch.isfinite(out).all()
    print("[OK] prefill 输出与缓存 (c_kv, k_rope) shape 正确")


def test_decode_paths_match_prefill():
    S = 12
    mla = MLA(CFG, max_seq_len=64, device=DEV, dtype=DT)
    h = torch.randn(1, S, CFG.d, device=DEV, dtype=DT)
    out, ckv, krope = mla.forward_prefill(h)

    # 用前 S-1 token 的缓存 decode 最后一个 token（position=S-1）
    h_t = h[0, S - 1]
    ckv_hist = ckv[0, : S - 1]
    krope_hist = krope[0, : S - 1]

    ref = out[0, S - 1]
    o_naive = mla.decode_naive(h_t, ckv_hist, krope_hist, S - 1)
    o_abs = mla.decode_absorbed(h_t, ckv_hist, krope_hist, S - 1)

    e_naive = frobenius_rel_err(o_naive, ref)
    e_abs_vs_ref = frobenius_rel_err(o_abs, ref)
    e_abs_vs_naive = frobenius_rel_err(o_abs, o_naive)
    print(f"[info] decode err vs prefill: naive={e_naive:.4f} absorbed={e_abs_vs_ref:.4f}")
    print(f"[info] absorbed vs naive: {e_abs_vs_naive:.4f}")
    assert e_naive < 0.05, f"naive decode 与 prefill 不一致: {e_naive}"
    assert e_abs_vs_ref < 0.05, f"absorbed decode 与 prefill 不一致: {e_abs_vs_ref}"
    assert e_abs_vs_naive < 0.05, f"absorbed 与 naive decode 不一致: {e_abs_vs_naive}"
    print("[OK] weight absorption 与 naive decode、prefill 三路径数值一致")


def test_compression_ratio():
    info = CFG.kv_bytes_per_token(2)
    assert abs(info["compression"] - 3.0) < 1e-6, info
    print(f"[OK] mini 配置 KV 压缩比={info['compression']:.2f}x "
          f"({info['mha_bytes']}B -> {info['mla_bytes']}B/token)")


if __name__ == "__main__":
    test_rope()
    test_prefill()
    test_decode_paths_match_prefill()
    test_compression_ratio()
    print("\n全部 MLA 测试通过 ✅")
