"""块池：跨块写入、填充、引用计数、前缀共享与 CoW。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cluster.kvpool.block_pool import BlockPool, SequenceKVMeta
from node_backend.deepseek_infer.model import mini_config

torch.manual_seed(0)
mcfg = mini_config()
pool = BlockPool(mcfg, "cpu", torch.float32)
L, dc, nh, dr = mcfg.n_layers, mcfg.d_latent, mcfg.n_heads, mcfg.d_head_rope
B = pool.B


def make_tokens(T, start=0):
    ckv = torch.zeros(L, T, dc)
    krope = torch.zeros(L, T, nh, dr)
    for t in range(T):
        ckv[:, t] = start + t
        krope[:, t, :, 0] = start + t
    return ckv, krope


def test_append_and_layout():
    meta = SequenceKVMeta("s1", pool)
    ckv, krope = make_tokens(40)
    pool.append_tokens(meta, ckv, krope)
    assert meta.length == 40
    assert sorted(meta.table) == [0, 1, 2]
    assert pool.blocks[meta.table[0]].fill == B
    assert pool.blocks[meta.table[2]].fill == 8
    # token 20 -> logical1 off4
    b = pool.blocks[meta.table[1]]
    assert torch.allclose(b.ckv[:, 4], ckv[:, 20])
    assert torch.allclose(b.krope[:, 4, :, 0], krope[:, 20, :, 0])
    print("[OK] 跨块写入/填充/定位正确")


def test_share_and_cow():
    src = SequenceKVMeta("src", pool)
    ckv, krope = make_tokens(40)
    pool.append_tokens(src, ckv, krope)

    # 共享前 2 个物理块，但逻辑长度只算 20（末块为部分共享）
    dst = SequenceKVMeta("dst", pool)
    pool.share_prefix(src, dst, n_blocks=2, shared_tokens=20)
    assert dst.length == 20
    assert pool.blocks[src.table[0]].refs == 2
    assert pool.blocks[src.table[1]].refs == 2

    # 再写 position 20（logical1 off4）-> 命中共享块 -> CoW
    nckv, nkr = make_tokens(1, start=999)
    pool.append_tokens(dst, nckv, nkr)
    assert dst.table[1] != src.table[1]          # 该块已复制
    assert dst.table[0] == src.table[0]          # 完整前缀块仍共享
    assert pool.blocks[src.table[1]].refs == 1   # 旧块回到 1
    cow = pool.blocks[dst.table[1]]
    assert torch.allclose(cow.ckv[:, 4], nckv[:, 0])
    print("[OK] 前缀共享 + 写时复制(CoW)正确")


def test_free():
    meta = SequenceKVMeta("free", pool)
    ckv, krope = make_tokens(20)
    pool.append_tokens(meta, ckv, krope)
    bids = set(meta.table.values())
    pool.free_sequence(meta)
    assert all(pool.blocks[b].refs == 0 for b in bids)
    print("[OK] 释放后引用归零")


if __name__ == "__main__":
    test_append_and_layout()
    test_share_and_cow()
    test_free()
    print("块池测试通过 ✅")
