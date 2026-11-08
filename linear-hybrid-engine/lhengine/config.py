# -*- coding: utf-8 -*-
"""
模型与引擎配置定义。

设计要点：
    1. MixerSpec 描述"某一层"的 mixer 类型与超参；
    2. ModelConfig 用一个 MixerSpec 列表表达整个网络，从而统一支持
       "纯 Mamba 堆叠"和"Hybrid（线性层 + 注意力层交替）"；
    3. 维度约定（全项目一致）：
         d_model : 残差流维度（每 token 的向量宽度）
         d_inner : mixer 内部扩展维度 = n_heads * d_head
         d_state : SSM 的状态维 N（Mamba1 常用 16，Mamba2 为 128）
         d_head  : 每头维度；DeltaNet 中 key 维 K 与 value 维 P 可分别指定
"""
from dataclasses import dataclass, field
from typing import List, Optional


# 支持的 mixer 类型字符串
MIXER_SSM = "mamba1"        # Mamba1：选择性 SSM，d_state 通常 16
MIXER_MAMBA2 = "mamba2"     # Mamba2：多头 SSD，d_state 通常 128
MIXER_DELTANET = "gdn"      # Gated DeltaNet：delta rule + 标量门控
MIXER_KDA = "kda"           # Kimi Delta Attention：channel-wise 门控
MIXER_SWA = "swa"           # Sliding Window Attention：滑动窗口注意力
MIXER_DELTA = "delta"       # 纯 DeltaNet（无门控）


@dataclass
class MixerSpec:
    """单层 mixer 的规格。"""
    kind: str                       # 类型，取上面的 MIXER_* 常量
    d_state: Optional[int] = None   # SSM 状态维；不指定则用模型默认
    window: int = 64                # SWA 的滑动窗口大小（仅 swa 用）

    @staticmethod
    def uniform(kind: str, n_layers: int) -> List["MixerSpec"]:
        """快速生成 n_layers 个同类型 mixer 的规格列表。"""
        return [MixerSpec(kind=kind) for _ in range(n_layers)]

    @staticmethod
    def hybrid(pattern: List[str]) -> List["MixerSpec"]:
        """按给定类型模式（如 ['gdn','gdn','gdn','swa']）生成规格列表。"""
        return [MixerSpec(kind=k) for k in pattern]


@dataclass
class ModelConfig:
    """语言模型配置。默认值对齐 mamba2-130m 的典型结构，便于加载真实权重。"""
    d_model: int = 768
    n_layers: int = 24
    vocab_size: int = 50280
    expand: int = 2                 # 内部扩展倍数，d_inner = expand * d_model
    d_head: int = 64                # 每头维度（Mamba2 为 64）
    d_state: int = 128              # SSM 状态维 N（Mamba2 为 128）
    d_key: int = 64                 # DeltaNet 状态的 key 维 K（Mamba2 系列中等于 d_head）
    conv_kernel: int = 4            # 因果 depthwise 短卷积核
    n_heads: Optional[int] = None   # 头数；None 时由 d_inner / d_head 自动推导
    n_kv_heads: Optional[int] = None  # SWA 中 GQA 的 kv 头数（None 表示不分组）
    rms_eps: float = 1e-5
    mixer_pattern: List[MixerSpec] = field(default_factory=list)
    # 当某层 mixer 未显式指定 d_state 时使用的默认值（mamba1 用 16）
    tie_gate_in_proj: bool = True   # in_proj 是否同时输出 gate z（Mamba 风格）

    def __post_init__(self):
        # 若未给 mixer 模式，默认全部为 mamba2
        if not self.mixer_pattern:
            self.mixer_pattern = MixerSpec.uniform(MIXER_MAMBA2, self.n_layers)
        # 层数与模式长度保持一致
        self.n_layers = len(self.mixer_pattern)

    @property
    def d_inner(self) -> int:
        """mixer 内部维度 = 扩展倍数 × 残差流维度。"""
        return self.expand * self.d_model

    @property
    def heads(self) -> int:
        """多头注意力/SSD 的头数，由内部维度与每头维度推导。"""
        if self.n_heads is not None:
            return self.n_heads
        assert self.d_inner % self.d_head == 0, "d_inner 必须能被 d_head 整除"
        return self.d_inner // self.d_head

    def state_dim_for(self, spec: MixerSpec) -> int:
        """某层实际使用的状态维：层规格优先，否则用模型默认。"""
        return spec.d_state if spec.d_state is not None else self.d_state


@dataclass
class EngineConfig:
    """生成引擎配置。"""
    chunk_size: int = 64            # chunkwise 内核的块大小（Mamba2 常用 64/256）
    use_cuda_kernels: bool = False  # 是否尝试加载已编译的 CUDA 内核
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    dtype: str = "float32"          # "float32" / "float16" / "bfloat16"
