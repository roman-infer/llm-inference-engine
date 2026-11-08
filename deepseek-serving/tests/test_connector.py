"""KV Connector：握手 + 数据经 loopback（序列化）收发一致。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cluster.comm.kv_connector import BlockData, KVConnector
from cluster.comm.transport import LoopbackTransport
from node_backend.deepseek_infer.model import mini_config

mcfg = mini_config()
L, dc, nh, dr = mcfg.n_layers, mcfg.d_latent, mcfg.n_heads, mcfg.d_head_rope


def test_roundtrip():
    tr = LoopbackTransport(serialize=True)
    tr.register("A")
    tr.register("B")
    conn = KVConnector()

    full = BlockData(0, torch.randn(L, 16, dc), torch.randn(L, 16, nh, dr), 16)
    part = BlockData(1, torch.randn(L, 5, dc), torch.randn(L, 5, nh, dr), 5)
    conn.send(tr, "A", "B", [full, part])

    assert tr.bytes_sent > 0                     # 真实走了字节编码
    got = conn.recv(tr, "A", "B")
    assert len(got) == 2
    assert got[0].fill == 16 and got[1].fill == 5
    assert torch.allclose(got[0].ckv, full.ckv)
    assert torch.allclose(got[1].krope, part.krope)
    print("[OK] KV 握手+序列化收发一致, wire bytes =", tr.bytes_sent)


if __name__ == "__main__":
    test_roundtrip()
    print("Connector 测试通过 ✅")
