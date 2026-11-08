# -*- coding: utf-8 -*-
"""
关联扫描（Associative / Prefix Scan）内核。

要并行化的递推：
    h_t = a_t * h_{t-1} + b_t
其中 a_t、b_t 是随时间变化的（时变），不能用卷积/FFT。

关键观察——二元算子的结合律：
    定义算子  (a, b) ∘ (c, d) = (a*c, a*d + b)
    则
      ((a,b)∘(c,d))∘(e,f) = (ac, ad+b)∘(e,f) = (ace, acf + ad + b)
      (a,b)∘((c,d)∘(e,f)) = (a,b)∘(ce, cf+d) = (ace, a(cf+d)+b) = (ace, acf+ad+b)
    两者相等 ⇒ 该算子满足结合律 ⇒ 可以像加法 cumsum 一样用树形/分块方式并行。
    这正是 GPU 并行 prefix scan（Blelloch scan）的代数基础。

本文件提供三种实现：
    naive_scan  纯循环参考实现（最清晰，用于数值对齐）
    block_scan  分块并行实现（块内循环、块间并行，接近 GPU 思路）
    scan        统一入口（默认 block_scan）
"""
import torch


def naive_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
    """
    循环版 prefix scan，逐元素递推。

    a, b : (..., L)，序列在最后一维
    h0   : (..., 1) 初始隐状态，None 表示零初始化
    return: (..., L) 每一步的隐状态
    """
    L = a.shape[-1]
    if h0 is None:
        h = torch.zeros_like(b[..., :1])
    else:
        h = h0
    outs = []
    for t in range(L):
        h = a[..., t:t + 1] * h + b[..., t:t + 1]
        outs.append(h)
    return torch.cat(outs, dim=-1)


def block_scan(a: torch.Tensor, b: torch.Tensor, block: int = 128) -> torch.Tensor:
    """
    分块并行 scan。

    思路（两级）：
      1. 块内：对每个块用循环算出块内每步结果，并把"整块"折叠成一个算子
               (A_block, B_block)，其中 A_block = ∏a，B_block 为合并后的 b；
      2. 块间：对各块的 (A_block, B_block) 做一次（规模小得多的）scan，
               得到每个块的前缀状态；
      3. 再把块前缀作用回各块内部结果。
    块数远小于序列长度，且块之间相互独立可并行，这是 GPU 上的标准做法。
    """
    *prefix, L = a.shape
    nb = (L + block - 1) // block  # 向上取整的块数

    # 末尾补"1（a）/0（b）"到整块长度，方便统一 reshape。
    # 注意：不能用 a[..., :pad] 再 ones_like——当 pad > L 时切片只返回 L 个；
    # 必须显式按 pad 构造补块的形状。
    pad = nb * block - L
    if pad:
        a_pad_shape = a.shape[:-1] + (pad,)
        a = torch.cat([a, torch.ones(a_pad_shape, dtype=a.dtype, device=a.device)], dim=-1)
        b_pad_shape = b.shape[:-1] + (pad,)
        b = torch.cat([b, torch.zeros(b_pad_shape, dtype=b.dtype, device=b.device)], dim=-1)

    # 重塑为 (..., nb, block)，把序列维拆成"块号 + 块内位置"
    a = a.reshape(*prefix, nb, block)
    b = b.reshape(*prefix, nb, block)

    # ---- 第 1 级：块内循环 scan ----
    h = torch.zeros_like(b[..., :1])  # 块内起始状态先按 0 算
    intra = []
    for r in range(block):
        h = a[..., r:r + 1] * h + b[..., r:r + 1]
        intra.append(h)
    intra = torch.cat(intra, dim=-1)  # (..., nb, block)

    # 把每个块折叠成单个算子 (A_block, B_block)
    # A_block = 块内所有 a 的乘积
    A_block = a.prod(dim=-1)          # (..., nb)
    # B_block = 块内从零状态出发走完整个块后的最终状态，即 intra 的最后一个
    B_block = intra[..., -1]          # (..., nb)

    # ---- 第 2 级：块间 scan（规模只有 nb）----
    block_prefix = naive_scan(A_block, B_block)  # (..., nb) 各块结束时的累计状态
    # 每个块的"起始前缀" = 上一个块的累计状态；第一个块为 0
    block_start = torch.zeros_like(block_prefix)
    block_start[..., 1:] = block_prefix[..., :-1]

    # ---- 第 3 级：把块前缀作用回块内 ----
    # 块内递推在非零起始状态 s 下：h_t = (∏_{1..t}a) * s + intra_t
    # 先算块内 a 的逐步步前缀乘积
    cum_a = torch.cumprod(a, dim=-1)  # (..., nb, block)
    out = cum_a * block_start.unsqueeze(-1) + intra

    # 拼回并去掉 padding
    out = out.reshape(*prefix, nb * block)
    return out[..., :L]


def scan(a: torch.Tensor, b: torch.Tensor, block: int = 128) -> torch.Tensor:
    """
    统一入口：时变线性递推的并行扫描。
    a, b : (..., L)；return (..., L)

    若已编译自研 CUDA 内核且输入在 CUDA 上，优先走 CUDA；否则用 PyTorch block_scan。
    """
    from .loader import cuda_ext
    ext = cuda_ext()
    if ext is not None and a.is_cuda and a.dtype == torch.float32:
        # 展平为 (C, L) 连续，调用 CUDA scan，再还原形状。
        lead = a.shape[:-1]
        L = a.shape[-1]
        a2 = a.reshape(-1, L).contiguous()
        b2 = b.reshape(-1, L).contiguous()
        y = ext.scan_forward(a2, b2)
        return y.reshape(*lead, L)
    return block_scan(a, b, block=block)
