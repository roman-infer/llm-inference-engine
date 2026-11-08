"""通信传输抽象：对齐 NCCL / RDMA 后端；本机用 LoopbackTransport 模拟跨节点链路。

真实部署后端（接口一致，替换即可）：
- NcclTransport : NCCL，P/D 位于同一 NVLink/IB 域；
- RdmaTransport : RDMA verbs / Mooncake TransferEngine，跨节点 KV 直传。
模拟后端：
- LoopbackTransport : 端点间消息队列；默认把 payload 序列化为字节再还原，
  真实覆盖"线上编解码"路径（serialize=False 可走零拷贝对象）。
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod

import torch


class Transport(ABC):
    @abstractmethod
    def register(self, endpoint: str) -> None: ...

    @abstractmethod
    def send(self, src: str, dst: str, payload, tag: str = "") -> None: ...

    @abstractmethod
    def recv(self, dst: str, src: str | None = None, tag: str = ""): ...


def _serialize(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _deserialize(blob: bytes):
    return torch.load(io.BytesIO(blob), weights_only=False)


class LoopbackTransport(Transport):
    """单进程内模拟一张消息网络：(src,dst,tag) 为队列键。"""

    def __init__(self, serialize: bool = True):
        self._endpoints: set[str] = set()
        self._mail: dict[tuple, list] = {}
        self.serialize = serialize
        self.bytes_sent = 0

    def register(self, endpoint: str) -> None:
        self._endpoints.add(endpoint)

    def send(self, src: str, dst: str, payload, tag: str = "") -> None:
        assert dst in self._endpoints, f"unknown dst endpoint: {dst}"
        if self.serialize:
            payload = _serialize(payload)
            self.bytes_sent += len(payload)
        self._mail.setdefault((src, dst, tag), []).append(payload)

    def recv(self, dst: str, src: str | None = None, tag: str = ""):
        if src is None:
            cand = [k for k, q in self._mail.items()
                    if k[1] == dst and k[2] == tag and q]
            if not cand:
                raise RuntimeError(f"no message for {dst} tag={tag}")
            key = cand[0]
        else:
            key = (src, dst, tag)
            if key not in self._mail or not self._mail[key]:
                raise RuntimeError(f"no message for {key}")
        blob = self._mail[key].pop(0)
        return _deserialize(blob) if self.serialize else blob

    def pending(self, dst: str, tag: str | None = None) -> int:
        return sum(len(q) for k, q in self._mail.items()
                   if k[1] == dst and (tag is None or k[2] == tag))
