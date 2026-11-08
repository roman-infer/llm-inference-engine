"""KV Connector：在 P/D（任意引擎）之间传输 KV 块，对齐 Mooncake / vLLM Connector。

带控制面握手：先发形状/元信息（handshake），再发张量（data），
接收方据此准备缓冲区——与分布式流水线"先发形状再发数据"采用同一模式。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockData:
    logical_index: int           # 该块在序列中的逻辑块序号
    ckv: torch.Tensor            # (L, B, d_latent)
    krope: torch.Tensor          # (L, B, n_heads, d_head_rope)
    fill: int                    # 块内有效 token 数（末块可能不满）


class KVConnector:
    HS_TAG = "kv_handshake"
    DATA_TAG = "kv_data"

    def send(self, transport, src: str, dst: str, blocks: list[BlockData]) -> None:
        meta = [
            {
                "li": b.logical_index,
                "fill": b.fill,
                "ckv_shape": tuple(b.ckv.shape),
                "krope_shape": tuple(b.krope.shape),
                "dtype": str(b.ckv.dtype),
            }
            for b in blocks
        ]
        transport.send(src, dst, {"blocks": meta}, self.HS_TAG)

        data = [
            {
                "li": b.logical_index,
                "fill": b.fill,
                "ckv": b.ckv.contiguous(),
                "krope": b.krope.contiguous(),
            }
            for b in blocks
        ]
        transport.send(src, dst, {"blocks": data}, self.DATA_TAG)

    def recv(self, transport, src: str, dst: str) -> list[BlockData]:
        transport.recv(dst, src, self.HS_TAG)      # 握手（元信息）
        data = transport.recv(dst, src, self.DATA_TAG)
        return [
            BlockData(item["li"], item["ckv"], item["krope"], item["fill"])
            for item in data["blocks"]
        ]
