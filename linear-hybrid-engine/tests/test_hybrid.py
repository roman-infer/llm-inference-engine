# -*- coding: utf-8 -*-
"""
Hybrid 端到端测试：
    1. 模型组装与随机初始化生成（字节 tokenizer，离线可跑）；
    2. prefill 整段与逐 token step 的 logits 对齐
       —— 验证 chunkwise prefill 与 recurrent decode 数学等价。
"""
import torch

from lhengine.config import ModelConfig, MixerSpec
from lhengine.models.lm import CausalLM
from lhengine.weights import random_init_
from lhengine.engine import Engine, sample_logits


def build_hybrid(pattern):
    """构造小尺寸 hybrid 模型（vocab 256，适配字节 tokenizer）。"""
    cfg = ModelConfig(
        d_model=64, n_layers=len(pattern), vocab_size=256,
        expand=2, d_head=16, d_state=16, d_key=16,
        mixer_pattern=MixerSpec.hybrid(pattern),
    )
    model = CausalLM(cfg)
    random_init_(model, seed=1)
    return model, cfg


def test_gdn_hybrid_generate():
    """3*gdn + 1*swa 混合模型可生成有限 token。"""
    model, cfg = build_hybrid(["gdn", "gdn", "gdn", "swa"])
    eng = Engine(model, tokenizer=None)
    ids = eng.generate_ids("hello", max_new_tokens=20, temperature=0.0)
    assert len(ids) > 5
    assert all(isinstance(i, int) and 0 <= i < cfg.vocab_size for i in ids)


def test_mamba2_hybrid_generate():
    """mamba2 与 swa 混合也可生成。"""
    model, cfg = build_hybrid(["mamba2", "swa", "mamba2", "swa"])
    eng = Engine(model, tokenizer=None)
    ids = eng.generate_ids("test", max_new_tokens=15, temperature=0.0)
    assert len(ids) > 5


def test_prefill_matches_step():
    """整段 prefill 的 logits 应与逐 token step 完全一致。"""
    model, _ = build_hybrid(["gdn", "gdn", "swa", "gdn"])
    model.eval()
    ids = torch.tensor([[10, 20, 30, 40, 50, 60]])

    # 整段 prefill，取所有位置 logits
    with torch.no_grad():
        all_logits, _ = model(ids, return_all_logits=True)  # (1,L,vocab)

    # 逐 token step：从空状态开始
    states = model.init_states(1, ids.device)
    step_logits = []
    for t in range(ids.shape[1]):
        with torch.no_grad():
            lg, states = model.step(ids[:, t:t + 1], states)
        step_logits.append(lg)
    step_logits = torch.stack(step_logits, dim=1)  # (1,L,vocab)

    assert torch.allclose(all_logits, step_logits, atol=1e-4, rtol=1e-4)


def test_greedy_generation_equivalence():
    """贪心生成：prefill+decode 路径产生的 token 应自洽（无异常值）。"""
    model, cfg = build_hybrid(["gdn", "swa", "gdn", "swa"])
    model.eval()
    ids = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        logits, states = model(ids)
    nxt = sample_logits(logits, temperature=0.0)
    assert nxt.ndim == 1 and 0 <= nxt.item() < cfg.vocab_size
