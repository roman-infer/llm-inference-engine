"""经 hf-mirror 下载 Qwen2.5-0.5B（真实语言模型，用于 MTP 端到端收益量化）。"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

path = snapshot_download(
    "Qwen/Qwen2.5-0.5B",
    allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                    "tokenizer_config.json", "vocab.json", "merges.txt",
                    "generation_config.json"],
)
print("MODEL_PATH:", path)
