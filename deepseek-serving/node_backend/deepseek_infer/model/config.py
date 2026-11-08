"""DeepSeek 风格模型配置 + mini / V2-Lite 预设。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    # MLA
    n_heads: int
    d_head: int
    d_head_rope: int
    d_latent: int
    d_latent_q: int
    # MoE
    moe_inter: int
    n_routed: int
    n_active: int
    n_shared: int
    # 首层 dense FFN
    dense_inter: int
    # misc
    max_seq_len: int = 4096
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    n_mtp: int = 1
    rms_eps: float = 1e-6
    initializer_range: float = 0.08


def mini_config() -> ModelConfig:
    """同构 mini DeepSeek（MLA+MoE+MTP），任何机器秒跑，用于端到端机制验证。"""
    return ModelConfig(
        vocab_size=8192, d_model=256, n_layers=4,
        n_heads=8, d_head=32, d_head_rope=16, d_latent=128, d_latent_q=128,
        moe_inter=128, n_routed=8, n_active=2, n_shared=2,
        dense_inter=512, max_seq_len=1024, tie_word_embeddings=True,
        rope_theta=10000.0, n_mtp=1,
    )


def deepseek_v2_lite_config() -> ModelConfig:
    """DeepSeek-V2-Lite 预设（15.7B/2.4B）。注意：真实加载时以 HF config.json 为准。"""
    return ModelConfig(
        vocab_size=102400, d_model=2048, n_layers=27,
        n_heads=16, d_head=128, d_head_rope=64, d_latent=512, d_latent_q=512,
        moe_inter=1408, n_routed=64, n_active=6, n_shared=2,
        dense_inter=10944, max_seq_len=32768, tie_word_embeddings=False,
        rope_theta=10000.0, n_mtp=1,
    )
