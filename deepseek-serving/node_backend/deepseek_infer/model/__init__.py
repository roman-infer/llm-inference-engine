"""DeepSeek 风格模型。"""
from .config import ModelConfig, mini_config, deepseek_v2_lite_config
from .norms import RMSNorm


def __getattr__(name):
    # 延迟导入 model.py，打破 model <-> moe 的循环依赖
    if name in ("DeepSeekModel", "Block", "DenseFFN"):
        from . import model as _model
        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ModelConfig", "mini_config", "deepseek_v2_lite_config", "DeepSeekModel",
           "Block", "DenseFFN", "RMSNorm"]
