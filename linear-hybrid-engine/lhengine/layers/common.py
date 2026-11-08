# -*- coding: utf-8 -*-
"""
通用组件：RMSNorm、因果 depthwise 短卷积、门控、线性投影。

这些组件在 Mamba 系列与 Hybrid 模型中都会用到：
    - RMSNorm        替代 LayerNorm，只做缩放不做均值偏移，计算更省
    - CausalConv1d   Mamba 块内的 depthwise 因果卷积（局部上下文混合），
                     论文中卷积核为 4，且必须左侧 padding（不能看到未来）
    - silu_gate      Mamba 块输出处的乘性门控：y = y * SiLU(z)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization）。

    公式：x_hat = x / sqrt(mean(x^2) + eps)，y = x_hat * weight
    与 LayerNorm 的区别：不减均值、只有一个可学习缩放向量。
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 权重初始化为 1，保证初始时归一化近似恒等
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 统计与归一化都在 float32 进行：x 为 fp16 时 x^2 容易上溢成 inf
        # （fp16 上限 ~65504），rsqrt(inf)=0 会把整维清零。最后再投回原 dtype。
        xf = x.float()
        norm = xf.pow(2).mean(dim=-1, keepdim=True)
        x_hat = xf * torch.rsqrt(norm + self.eps)
        return (self.weight.float() * x_hat).to(x.dtype)


class CausalConv1d(nn.Module):
    """
    因果 depthwise 一维卷积。

    - in_channels == out_channels，每个通道独立卷积（depthwise）
    - 左侧 padding kernel_size-1，卷积后裁掉右侧，保证 t 时刻只看到 <=t 的输入
    - Mamba 论文中该卷积用于在 SSM 之前做局部平滑，核大小为 4
    """

    def __init__(self, d_inner: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.d_inner = d_inner
        # conv1d 权重形状 (out_channels, in_channels/groups, kernel_size)
        self.conv = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=kernel_size,
            groups=d_inner,      # depthwise：通道互不混合
            padding=0,           # 手动做因果 padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C) -> 输出 (B, L, C)
        """
        B, L, C = x.shape
        # 左侧 padding：在序列维前面补 kernel_size-1 个零
        x = x.transpose(1, 2)  # -> (B, C, L)
        x = F.pad(x, (self.kernel_size - 1, 0))  # 只 pad 左边
        x = self.conv(x)
        # 卷积后长度为 L+kernel_size-1，取前 L 个即严格因果
        x = x[:, :, :L]
        return x.transpose(1, 2)  # -> (B, L, C)

    def step(self, x_t: torch.Tensor, conv_state: torch.Tensor):
        """
        decode 单步版本：用一个滑动窗口缓存做卷积，避免每步重算整段。

        x_t        : (B, C) 当前 token
        conv_state : (B, C, kernel_size-1) 之前 kernel_size-1 个输入
        return     : (输出 (B,C), 新的 conv_state)
        """
        k = self.kernel_size
        # 把当前输入拼到历史窗口后：[旧 k-1 个, 当前 1 个]
        full = torch.cat([conv_state, x_t.unsqueeze(-1)], dim=-1)  # (B,C,k)
        weight = self.conv.weight[:, 0, :]  # (C, k)
        # 逐通道加权求和（depthwise）
        out = (full * weight.unsqueeze(0)).sum(dim=-1)  # (B,C)
        if self.conv.bias is not None:
            out = out + self.conv.bias
        # 新窗口丢掉最旧的一个
        new_state = full[:, :, 1:]
        return out, new_state


def silu_gate(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Mamba 块输出门控：y * SiLU(z)。
    z 是 in_proj 旁路产生的门控分支，SiLU 后与主干逐元素相乘。
    """
    return y * F.silu(z)
