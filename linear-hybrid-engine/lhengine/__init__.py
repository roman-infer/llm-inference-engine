# -*- coding: utf-8 -*-
"""
linear-hybrid-engine（后 Transformer 线性/混合架构推理引擎）

子模块：
    config      模型与引擎配置
    layers      各类 mixer 层（Mamba1/Mamba2/DeltaNet/GatedDeltaNet/KDA/SWA）
    models      语言模型组装（纯线性 / Hybrid）
    cache       固定大小 state 缓存、hybrid 双缓存
    engine      生成引擎（prefill / decode 主循环）
    weights     safetensors 权重加载
    tokenizer   tokenizer 封装
    kernels     高性能内核（PyTorch chunkwise + 可选 CUDA）
"""
from .config import ModelConfig, MixerSpec, EngineConfig

__all__ = ["ModelConfig", "MixerSpec", "EngineConfig"]
__version__ = "0.1.0"
