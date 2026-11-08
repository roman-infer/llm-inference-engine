# -*- coding: utf-8 -*-
"""
因果语言模型（CausalLM）组装。

结构（pre-norm 残差）：
    token -> embedding
    重复 n_layers 次：x = x + Mixer_i(RMSNorm_i(x))
    x = RMSNorm_final(x)
    logits = lm_head(x)
通过 cfg.mixer_pattern 决定每层是哪一种 mixer，从而同时支持：
    - 纯线性堆叠（如全 mamba2）
    - Hybrid（如 3 个 gdn + 1 个 swa 交替）
"""
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..layers import RMSNorm, MixerBlock


class CausalLM(nn.Module):
    """后 Transformer 因果语言模型。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # token embedding（不做额外缩放，与 Mamba 权重一致）
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # 每层一个输入 RMSNorm 与一个 mixer
        self.norms = nn.ModuleList(
            [RMSNorm(cfg.d_model, eps=cfg.rms_eps) for _ in range(cfg.n_layers)])
        self.mixers = nn.ModuleList(
            [MixerBlock(spec, cfg) for spec in cfg.mixer_pattern])
        # 最终 RMSNorm
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        # 输出投影（权重可与 embedding 绑定，加载时处理）
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, states=None,
                return_all_logits: bool = False):
        """
        整段前向（prefill）。
        input_ids: (b, L)
        states   : 可选初始状态列表（续接上下文）
        return: (logits, new_states)
                 logits 默认只取最后位置 (b, vocab)；return_all_logits 时取全部 (b,L,vocab)
        """
        x = self.embedding(input_ids)
        new_states = []
        for i, (norm, mixer) in enumerate(zip(self.norms, self.mixers)):
            prev = states[i] if states is not None else None
            h = norm(x)
            delta, st = mixer(h, state=prev)
            x = x + delta
            new_states.append(st)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if not return_all_logits:
            logits = logits[:, -1]
        return logits, new_states

    @torch.no_grad()
    def step(self, input_id: torch.Tensor, states: list):
        """
        decode 单步。
        input_id: (b, 1)
        states  : 上一步的状态列表
        return: (logits (b,vocab), new_states)
        """
        x = self.embedding(input_id)  # (b,1,d)
        new_states = []
        for i, (norm, mixer) in enumerate(zip(self.norms, self.mixers)):
            h = norm(x)
            # 单步 mixer 接口：输入 (b,d)，输出 (b,d)
            delta, st = mixer.step(h.squeeze(1), states[i])
            x = x + delta.unsqueeze(1)
            new_states.append(st)
        x = self.norm_f(x)
        logits = self.lm_head(x[:, -1])
        return logits, new_states

    def init_states(self, batch: int, device, dtype=torch.float32):
        """初始化所有层的 decode 状态。"""
        return [mixer.init_state(batch, device, dtype=dtype) for mixer in self.mixers]
