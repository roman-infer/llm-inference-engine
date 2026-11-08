# -*- coding: utf-8 -*-
"""
最小可运行示例：离线（无需下载权重）体验 linear-hybrid-engine 的核心能力。

运行（在项目根目录）：
    python examples/demo.py

内容：
    1. 构造一个 Hybrid 模型（GatedDeltaNet + 滑动窗口注意力交替）；
    2. prefill 得到固定大小 state；
    3. 逐 token decode 生成；
    4. 用 StateCache 演示前缀分支（fork）——状态是紧凑定长矩阵，复制成本极低。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lhengine.config import ModelConfig, MixerSpec
from lhengine.models.lm import CausalLM
from lhengine.weights import random_init_
from lhengine.cache import StateCache, clone_states


def build():
    """构造小尺寸 Hybrid 模型（vocab 256，字节 tokenizer）。"""
    cfg = ModelConfig(
        d_model=64, n_layers=4, vocab_size=256, expand=2,
        d_head=16, d_state=16, d_key=16,
        # 3 个线性层 + 1 个滑窗注意力层的混合模式。
        mixer_pattern=MixerSpec.hybrid(["gdn", "gdn", "swa", "gdn"]),
    )
    model = CausalLM(cfg)
    random_init_(model, seed=1)
    model.eval()
    return model, cfg


def main():
    model, cfg = build()

    # ---- 1) prefill：整段前向，得到各层固定状态 ----
    prompt_ids = torch.tensor([[10, 20, 30, 40, 50]])
    with torch.no_grad():
        logits, states = model(prompt_ids)
    print("prefill 完成，共", len(states), "层状态。")
    for i, s in enumerate(states):
        kind = type(s).__name__
        print(f"  层{i}: 状态类型={kind}")

    # ---- 2) decode：逐 token 生成（复用固定状态）----
    n_new = 8
    cur = logits.argmax(dim=-1, keepdim=True)  # 贪心取第一个新 token
    for _ in range(n_new):
        with torch.no_grad():
            logits, states = model.step(cur, states)
        cur = logits.argmax(dim=-1, keepdim=True)
        print("decode token:", cur.item())

    # ---- 3) StateCache：前缀分支（fork）----
    cache = StateCache()
    cache.put("seq-a", states)
    # 从 seq-a 分叉出 seq-b：深拷贝紧凑状态（无页表/引用计数）。
    ok = cache.fork("seq-a", "seq-b")
    print("fork 成功:", ok, "；缓存中序列数:", len(cache))

    # 也可以直接克隆一组状态用于并行采样。
    cloned = clone_states(states)
    print("克隆状态层数:", len(cloned))


if __name__ == "__main__":
    main()
