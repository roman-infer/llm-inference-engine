# -*- coding: utf-8 -*-
"""
权重加载：safetensors 真实权重 / 随机合成权重。

支持两类来源：
    1. 真实 mamba2 HuggingFace 权重（state-spaces/mamba2-*）：
       把 HF 的参数名（backbones.layers.i.mixer.*）映射到本模型命名
       （mixers.i.impl.*），逐张量加载并校验形状；
    2. 合成权重：随机初始化（离线测试/demo 用），保证项目不联网也能跑。
"""
import glob
import os
import re
import torch
import torch.nn as nn
from safetensors.torch import load_file

from .models.lm import CausalLM


def map_hf_key(key: str) -> str:
    """
    把 mamba2 HF 权重的参数名转换为本模型参数名。

    规则（一一对应）：
        backbones.embedding.weight            -> embedding.weight
        backbones.layers.i.norm.weight        -> norms.i.weight
        backbones.layers.i.mixer.*            -> mixers.i.impl.*
        mixer.conv1d.*                        -> impl.conv.conv.*
        backbones.norm_f.weight               -> norm_f.weight
        lm_head.weight                        -> lm_head.weight
    """
    if key == "backbones.embedding.weight":
        return "embedding.weight"
    if key == "backbones.norm_f.weight":
        return "norm_f.weight"
    m = re.match(r"backbones\.layers\.(\d+)\.norm\.weight", key)
    if m:
        return f"norms.{m.group(1)}.weight"
    m = re.match(r"backbones\.layers\.(\d+)\.mixer\.(.+)", key)
    if m:
        idx, rest = m.group(1), m.group(2)
        rest = rest.replace("conv1d.", "conv.conv.")
        return f"mixers.{idx}.impl.{rest}"
    if key == "lm_head.weight":
        return "lm_head.weight"
    return key  # 已是本模型命名或未知，原样返回


@torch.no_grad()
def load_safetensors_into(model: CausalLM, st_paths) -> dict:
    """
    把一个或多个 safetensors 文件加载进模型。

    st_paths: 文件路径或路径列表
    return: 加载报告 {"loaded": n, "shape_mismatch": [...], "missing": [...]}
    """
    if isinstance(st_paths, str):
        st_paths = [st_paths]
    state = {}
    for p in st_paths:
        state.update(load_file(p))

    # 先做 key 映射
    mapped = {map_hf_key(k): v for k, v in state.items()}
    own = dict(model.named_parameters())
    report = {"loaded": 0, "shape_mismatch": [], "unknown": [], "tied": False}
    for k, v in mapped.items():
        if k not in own:
            report["unknown"].append(k)
            continue
        if own[k].shape != v.shape:
            report["shape_mismatch"].append((k, tuple(own[k].shape), tuple(v.shape)))
            continue
        own[k].data.copy_(v.to(own[k].device).to(own[k].dtype))
        report["loaded"] += 1

    # 若没有 lm_head 权重，则与 embedding 绑定（Mamba 小模型常 tie）
    if "lm_head.weight" not in mapped:
        model.lm_head.weight = model.embedding.weight
        report["tied"] = True

    # ---- fail fast：绝不静默退化成随机初始化 ----
    # 本模型把 HF mamba2 合并的 in_proj（[z|x|B|C|dt]）拆成了独立的
    # in_proj / B_proj / C_proj / dt_proj，命名与切分都和 HF 不完全一致。
    # 若几乎没加载到权重就继续跑，用户会看到"加载完成"的日志却拿到
    # 随机出词——这是最严重的失败模式。这里直接报错，提示需要做权重转换。
    if report["loaded"] == 0:
        raise RuntimeError(
            "load_safetensors_into: 没有任何权重成功加载（loaded=0）。\n"
            f"  unknown keys（前 5 个）: {report['unknown'][:5]}\n"
            "  这通常意味着传入的是 HF mamba2 原始权重，其合并的 in_proj / "
            "dt_bias / A_log / mixer.norm 需要先按本模型的拆分方式转换后才能加载；\n"
            "  在转换脚本就绪前，请改用 random_init_() 跑 demo，避免将真实权重路径误认为可以加载。")
    if report["shape_mismatch"]:
        ex = report["shape_mismatch"][0]
        raise RuntimeError(
            "load_safetensors_into: 权重形状不匹配 "
            f"{len(report['shape_mismatch'])} 处，首例 key={ex[0]} "
            f"model_shape={ex[1]} ckpt_shape={ex[2]}。\n"
            "  同样提示：需要先写 HF->本模型的权重转换（切分 in_proj、映射 "
            "dt_bias/A_log/mixer.norm），不要带着不匹配的权重继续前向。")
    return report


def random_init_(model: CausalLM, seed: int = 0):
    """
    随机初始化（合成权重），用于离线测试与 demo。
    小尺度正态/均匀初始化，保证前向数值有限；embedding 与 lm_head 绑定。
    """
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    # 归一化缩放保持 1
    for m in model.modules():
        if isinstance(m, nn.LayerNorm) or type(m).__name__ == "RMSNorm":
            nn.init.ones_(m.weight)
    model.lm_head.weight = model.embedding.weight
    return model


def find_safetensors(directory: str):
    """在目录下查找所有 .safetensors 文件（含子目录）。"""
    return sorted(glob.glob(os.path.join(directory, "**", "*.safetensors"),
                            recursive=True))
