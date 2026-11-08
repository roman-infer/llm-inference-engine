# -*- coding: utf-8 -*-
"""
Tokenizer 封装。

两种工作方式：
    1. 有 HF tokenizer 目录时：加载真实 tokenizer（与模型权重配套）；
    2. 离线 fallback：字节级 tokenizer（每个 token 是一个字节，0 为 eos，
       字节 b 映射到 id=b+1，即 id 范围 1..256），保证不下载任何文件也能
       跑通 demo；因此要求模型 vocab_size >= 257（不是 256），否则字节
       0xFF 映射出的 id=256 会越界 embedding。
"""
import torch


class ByteTokenizer:
    """字节级 fallback tokenizer：文本 <-> 字节 id。"""

    pad_id = 0
    eos_id = 0
    # id = 字节 + 1，最大到 256；模型词表必须 >= 257。
    max_id = 256

    def encode(self, text: str):
        # UTF-8 编码后逐字节取 id，并加 1 以避开 0（保留 0 作为 eos）
        ids = [b + 1 for b in text.encode("utf-8")]
        assert max(ids, default=0) <= self.max_id, "字节 id 越界"
        return ids

    def decode(self, ids):
        # 去掉 0，其余 id 减 1 还原字节，再 UTF-8 解码（容错非法字节）
        bs = bytes([i - 1 for i in ids if i > 0])
        return bs.decode("utf-8", errors="replace")


def load_tokenizer(path: str = None):
    """
    加载 tokenizer：path 为 HF tokenizer 目录；None 或加载失败时用字节级。
    """
    if path is not None:
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(path)
            return HFTokenizerAdapter(tok)
        except Exception:
            pass
    return ByteTokenizer()


class HFTokenizerAdapter:
    """对 HF tokenizers.Tokenizer 的轻量适配，统一 encode/decode 接口。"""

    def __init__(self, tok):
        self.tok = tok
        self.eos_id = tok.token_to_id("<|endoftext|>")

    def encode(self, text: str):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(list(ids))


def encode_batch(tokenizer, text, device):
    """把文本编码为 (1, L) 的 LongTensor。"""
    ids = tokenizer.encode(text)
    return torch.tensor([ids], dtype=torch.long, device=device)
