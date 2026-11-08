"""环境核查：确认 GPU、FP8 GEMM、CUDA Graph、Triton 等关键能力。"""
import json
import platform
import torch

HAS_TRITON = False
triton_version = None
try:
    import triton
    HAS_TRITON = True
    triton_version = triton.__version__
except Exception:
    pass


def check_scaled_mm(blockwise: bool):
    m, k, n = 256, 512, 256
    a = torch.randn(m, k, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(n, k, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    if blockwise:
        # RowWise: 激活每行一个 scale, 权重每输出列一个 scale
        sa = torch.ones(m, 1, dtype=torch.float32, device="cuda")
        sw = torch.ones(1, n, dtype=torch.float32, device="cuda")
    else:
        sa = torch.ones(1, dtype=torch.float32, device="cuda")
        sw = torch.ones(1, dtype=torch.float32, device="cuda")
    try:
        r = torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sw, out_dtype=torch.float16)
        return {"ok": True, "shape": list(r.shape), "dtype": str(r.dtype)}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:300]}


def check_cuda_graph():
    try:
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        w = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                y = x @ w
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g, stream=s):
            y = x @ w
        g.replay()
        torch.cuda.synchronize()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:300]}


def main():
    prop = torch.cuda.get_device_properties(0)
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_built": torch.version.cuda,
        "device": prop.name,
        "capability": f"{prop.major}.{prop.minor}",
        "total_vram_GB": round(prop.total_memory / 1e9, 2),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "fp8_dtypes": {
            "e4m3": hasattr(torch, "float8_e4m3fn"),
            "e5m2": hasattr(torch, "float8_e5m2"),
        },
        "triton": {"available": HAS_TRITON, "version": triton_version},
        "scaled_mm_per_tensor": check_scaled_mm(False),
        "scaled_mm_blockwise": check_scaled_mm(True),
        "cuda_graph": check_cuda_graph(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
