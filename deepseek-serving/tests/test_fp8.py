"""FP8 量化与三种 Linear 路径的数值正确性测试。"""
import torch

from common import frobenius_rel_err, signal_rel_err, cosine_sim
from node_backend.deepseek_infer.fp8 import FP8Linear, quantize as Q

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_blockwise_quant_dequant():
    w = torch.randn(128, 512, device=DEV, dtype=torch.float16)
    wq, scales = Q.blockwise_weight_quant(w, block_size=128)
    assert wq.shape == w.shape and scales.shape == (4,)
    # 分块反量化
    wq_b = wq.reshape(128, 4, 128).float()
    deq = (wq_b * scales.reshape(1, -1, 1)).reshape(128, 512)
    err = frobenius_rel_err(deq, w)
    assert err < 0.03, f"blockwise 量化反量化相对误差过大: {err}"
    print(f"[OK] blockwise 量化/反量化 Frobenius err={err:.4f}")


def test_tokenwise_quant():
    x = torch.randn(64, 512, device=DEV, dtype=torch.float16) * torch.randint(1, 8, (64, 1), device=DEV)
    xq, sa = Q.tokenwise_quant(x)
    assert sa.shape == (64, 1)
    deq = xq.float() * sa
    err = frobenius_rel_err(deq, x)
    assert err < 0.03, f"per-token 量化误差过大: {err}"
    print(f"[OK] per-token 量化/反量化 Frobenius err={err:.4f}")


def test_linear_paths():
    m, k, n = 64, 512, 128
    lin = FP8Linear(k, n, block_size=128, device=DEV)
    x = torch.randn(m, k, device=DEV, dtype=torch.float16)
    ref = lin.forward_fp16(x)
    out_t = lin.forward_tensorwise(x)
    out_b = lin.forward_blockwise(x)

    err_t = frobenius_rel_err(out_t, ref)
    err_b = frobenius_rel_err(out_b, ref)
    sig_b = signal_rel_err(out_b, ref)
    cos_b = cosine_sim(out_b, ref)
    print(f"[info] Frobenius err: tensorwise={err_t:.4f} | blockwise={err_b:.4f}")
    print(f"[info] blockwise 信号位 rel err={sig_b:.4f}, cosine={cos_b:.4f}")
    assert err_t < 0.12, f"tensorwise 路径误差过大: {err_t}"
    assert err_b < 0.06, f"blockwise 路径误差过大: {err_b}"
    assert cos_b > 0.99, f"方向一致性过低: {cos_b}"
    # 同分布权重下两者精度接近（blockwise 不更差即可）
    assert err_b <= err_t + 0.01, f"blockwise 异常劣化: {err_b} vs {err_t}"
    print("[OK] FP8 Linear 三路径数值正确")


def test_outlier_isolation():
    """blockwise 的真实价值：单个 block 的极端 outlier/溢出不污染其余 block。

    FP8 是浮点格式，正常分布下 per-tensor 与 blockwise 相对精度近似恒定、无显著差异；
    blockwise 提供的是故障隔离边界（这也是 DeepSeek 采用细粒度 block 的工程动机之一）。
    """
    n, k = 128, 512
    torch.manual_seed(0)
    w = torch.randn(n, k, device=DEV, dtype=torch.float16)
    w[0, 0] = 100000.0  # fp16 中接近/达到溢出，位于 block 0

    wq_t, sw_t = Q.per_tensor_quant(w)
    deq_t = wq_t.float() * sw_t
    wq_b, sc_b = Q.blockwise_weight_quant(w, 128)
    deq_b = (wq_b.reshape(n, 4, 128).float() * sc_b.reshape(1, 4, 1)).reshape(n, k)

    ok_t = torch.isfinite(deq_t[:, 128:]).all().item()
    err_b = frobenius_rel_err(deq_b[:, 128:], w[:, 128:])
    print(f"[info] 极端 outlier: per-tensor 全局有限={ok_t} sw={sw_t.item()} | "
          f"blockwise 小block err={err_b:.4f}")
    assert not ok_t, "per-tensor 应被 outlier 污染（scale inf）"
    assert err_b < 0.05, f"blockwise 应隔离故障，小 block 保持正常: {err_b}"
    print("[OK] blockwise outlier 故障隔离成立，其余 block 精度不受影响")


def test_bias_shapes():
    lin = FP8Linear(256, 64, block_size=128, device=DEV)
    x = torch.randn(8, 256, device=DEV, dtype=torch.float16)
    for mode in ("fp16", "tensorwise", "blockwise"):
        y = lin(x, mode=mode)
        assert y.shape == (8, 64), f"{mode} shape 错误"
    print("[OK] 三种路径输出 shape 正确（含非整除 pad 路径）")


if __name__ == "__main__":
    test_blockwise_quant_dequant()
    test_tokenwise_quant()
    test_linear_paths()
    test_outlier_isolation()
    test_bias_shapes()
    print("\n全部 FP8 测试通过 ✅")

