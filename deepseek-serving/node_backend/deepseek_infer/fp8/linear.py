from __future__ import annotations

import torch

from . import quantize as Q


class FP8Linear(torch.nn.Module):
    """FP8 线性层，提供三条前向路径。

    - forward_fp16:       fp16 参考（x @ W^T + bias）
    - forward_tensorwise: per-tensor 量化；CUDA 走 FP8 tensor core（torch._scaled_mm），
                          无 tensor core 时回退为“量化->反量化->matmul”的数值等价模拟
    - forward_blockwise:  DeepSeek 口径 1x128 block 量化，K 维块循环严格等价
                          C = sa · Σ_j sw_j · (xq_j @ wq_j^T)
    """

    def __init__(self, in_features: int, out_features: int, block_size: int = 128,
                 bias: bool = True, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        # CPU-only torch 上自动回退，避免在没有 CUDA 时构造 cuda 张量
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1.0 / (self.in_features ** 0.5) if self.in_features > 0 else 0.0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # ---------------- fp16 参考 ----------------
    def forward_fp16(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return torch.nn.functional.linear(x, w, b)

    # ---------------- per-tensor / tensorwise ----------------
    def forward_tensorwise(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        xq, sa = Q.per_tensor_quant(x)          # sa: 标量
        wq, sw = Q.per_tensor_quant(w)          # sw: 标量

        out = None
        if x.is_cuda and w.is_cuda:
            try:
                # a:(M,K)fp8, b:(K,N)fp8；scale_a/scale_b 在内部应用
                out = torch._scaled_mm(
                    xq, wq.t().contiguous(),
                    scale_a=sa.reshape(()), scale_b=sw.reshape(()),
                    out_dtype=torch.float16)
            except Exception:
                out = None  # 任何不兼容都回退到数值等价的模拟路径
        if out is None:
            out = (xq.float() * sa.float()) @ (wq.float() * sw.float()).t()
            out = out.to(x.dtype)

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.to(x.dtype)

    # ---------------- 1x128 blockwise ----------------
    def forward_blockwise(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        B = self.block_size
        x_pad, _ = Q.pad_k(x, B)               # (M, Kp)，Kp 为 B 整数倍
        w_pad, _ = Q.pad_k(w, B)               # (N, Kp)
        M, Kp = x_pad.shape
        N = w_pad.shape[0]
        G = Kp // B

        xq, sa = Q.tokenwise_quant(x_pad)            # 激活 per-token: sa (M,1)
        wq, sw = Q.blockwise_weight_quant(w_pad, B)  # 权重 1x128: sw (G,)

        # 把 scale 吸收进对应块：sa 沿 token、sw 沿 K-block
        xb = xq.reshape(M, G, B).float() * sa.float().reshape(M, 1, 1)
        wb = wq.reshape(N, G, B).float() * sw.float().reshape(1, G, 1)

        acc = torch.zeros(M, N, device=x.device, dtype=torch.float32)
        for j in range(G):
            acc += xb[:, j, :] @ wb[:, j, :].t()

        if self.bias is not None:
            acc = acc + self.bias.float()
        return acc.to(x.dtype)

    # ---------------- 统一分发 ----------------
    def forward(self, x: torch.Tensor, mode: str = "fp16") -> torch.Tensor:
        if mode == "fp16":
            return self.forward_fp16(x)
        if mode == "tensorwise":
            return self.forward_tensorwise(x)
        if mode == "blockwise":
            return self.forward_blockwise(x)
        raise ValueError(f"未知 mode: {mode}（可选 fp16/tensorwise/blockwise）")
