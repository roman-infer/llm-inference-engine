"""FP8 量化与 FP8 Linear。"""
from .linear import FP8Linear
from . import quantize

__all__ = ["FP8Linear", "quantize"]
