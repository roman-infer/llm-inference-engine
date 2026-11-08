# -*- coding: utf-8 -*-
"""
下载真实 Mamba2 权重（HuggingFace），供本引擎加载生成。

用法（在项目根目录）：
    python scripts/download_weights.py --model state-spaces/mamba2-130m --out weights/mamba2-130m
    python scripts/download_weights.py --model state-spaces/mamba2-370m
    python scripts/download_weights.py --model state-spaces/mamba2-1.4b

说明：
    - 会下载 safetensors 权重、config.json、tokenizer 文件到 --out 目录；
    - 单卡 8GB 显存建议从 130m / 370m 起步（CPU 也能跑，只是慢）；
    - 若网络访问 HuggingFace 受限，可设置镜像：
          set HF_ENDPOINT=https://hf-mirror.com   (Windows)
"""
import argparse
import os
import sys

# 允许从项目根直接运行脚本：把项目根加入模块搜索路径。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="下载 Mamba2 权重")
    parser.add_argument("--model", default="state-spaces/mamba2-130m",
                        help="HuggingFace 模型仓库名")
    parser.add_argument("--out", default=None,
                        help="本地保存目录（默认 weights/<仓库名末段>）")
    args = parser.parse_args()

    # 允许通过环境变量切换 HF 镜像。
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        print(f"使用 HF_ENDPOINT={endpoint}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少 huggingface_hub，请先安装：pip install huggingface_hub")
        sys.exit(1)

    # 默认保存到项目根下的 weights/<repo 末段>。
    repo_name = args.model.split("/")[-1]
    out = args.out or os.path.join("weights", repo_name)

    print(f"开始下载 {args.model} -> {out}")
    path = snapshot_download(
        repo_id=args.model,
        local_dir=out,
        # 只下载推理所需文件，跳过无用的 pt 原始权重（优先 safetensors）。
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer.model"],
    )
    print(f"下载完成，权重位于：{path}")
    print("接下来可运行：python scripts/run_generation.py --weights", out,
          '--prompt "Hello"')


if __name__ == "__main__":
    main()
