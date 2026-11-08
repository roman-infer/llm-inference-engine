"""节点工厂：以同一 seed 构建节点后端模型 + latent KV cache。"""
from __future__ import annotations

import torch

from node_backend.deepseek_infer.engine.kv_cache import LatentKVCache
from node_backend.deepseek_infer.model import DeepSeekModel


def build_node(mcfg, seed: int, device: str, dtype):
    torch.manual_seed(seed)
    model = DeepSeekModel(mcfg, device=device, dtype=dtype)
    cache = LatentKVCache(mcfg, device=device, dtype=dtype)
    model.cache = cache
    return model, cache
