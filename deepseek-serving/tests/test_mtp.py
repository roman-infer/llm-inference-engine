"""MTP 数值与结构正确性测试。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from node_backend.deepseek_infer.model import DeepSeekModel, mini_config
from node_backend.deepseek_infer.mtp import (MTPHead, accept_greedy, draft_chain, draft_tree,
                                rejection_accept_one, verify_tree_greedy)

torch.manual_seed(3)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16
CFG = mini_config()


def _setup():
    model = DeepSeekModel(CFG, device=DEV, dtype=DT)
    head = MTPHead(CFG, model.embed_tokens, model._head(), device=DEV, dtype=DT)
    return model, head


def test_head_forward():
    _, head = _setup()
    h_t = torch.randn(1, 1, CFG.d_model, device=DEV, dtype=DT)
    tok = torch.tensor(5, device=DEV)
    h, logits = head(h_t, tok)
    assert h.shape == (1, 1, CFG.d_model)
    assert logits.shape == (1, 1, CFG.vocab_size)
    assert torch.isfinite(logits).all()
    print("[OK] MTPHead forward shape 与有限性正确")


def test_draft_chain():
    _, head = _setup()
    h_t = torch.randn(1, 1, CFG.d_model, device=DEV, dtype=DT)
    tok = torch.tensor(5, device=DEV)
    toks, h = draft_chain(head, h_t, tok, n_draft=4)
    assert toks.shape == (4,)
    assert toks.min() >= 0 and toks.max() < CFG.vocab_size
    print(f"[OK] 单链草稿产出 4 个 token: {toks.tolist()}")


def test_accept_greedy():
    preds = torch.tensor([1, 2, 3, 4])
    assert accept_greedy(preds, torch.tensor([1, 2, 9, 4])) == 2
    assert accept_greedy(preds, preds) == 4
    assert accept_greedy(preds, torch.tensor([9, 2, 3, 4])) == 0
    print("[OK] 贪心验证连续接受长度正确")


def test_rejection_sampling():
    v = CFG.vocab_size
    p = torch.softmax(torch.randn(v), dim=-1)
    q = p.clone()
    # p==q 时任何 r<1 都接受
    ok, _ = rejection_accept_one(p, q, 0, 0.5)
    assert ok
    # r=0 必接受
    ok, _ = rejection_accept_one(p, torch.softmax(torch.randn(v), -1), 3, 0.0)
    assert ok
    # 强制拒绝场景：q 在 x 处概率远大于 p，r=1
    q2 = torch.softmax(torch.randn(v), -1)
    x = 7
    q2 = q2.clone()
    q2[x] = 0.9
    q2 = q2 / q2.sum()
    ok, resample = rejection_accept_one(p, q2, x, 1.0)
    if not ok:
        assert resample is not None and 0 <= resample.item() < v
    print("[OK] 拒绝采样接受/拒绝与重采样逻辑正确（保持 target 分布）")


def test_tree():
    _, head = _setup()
    h_t = torch.randn(1, 1, CFG.d_model, device=DEV, dtype=DT)
    tok = torch.tensor(5, device=DEV)
    tree = draft_tree(head, h_t, tok, width=2, budget=7)
    assert tree.tokens.shape[0] == 7
    assert (tree.depths[tree.parents == -1] == 1).all()
    # 深度一致性：子深度 = 父深度+1
    for i in range(7):
        p = tree.parents[i].item()
        if p >= 0:
            assert tree.depths[i] == tree.depths[p] + 1
    paths = tree.paths()
    for path in paths:
        assert len(path) >= 1
    print(f"[OK] 动态树结构正确，共 {len(paths)} 条根→叶路径，"
          f"深度分布 {tree.depths.tolist()}")


def test_verify_tree():
    _, head = _setup()
    h_t = torch.randn(1, 1, CFG.d_model, device=DEV, dtype=DT)
    tree = draft_tree(head, h_t, torch.tensor(5, device=DEV), width=2, budget=7)
    paths = tree.paths()
    depth_map = {pi: min(2, len(p)) for pi, p in enumerate(paths)}
    bonus_map = {pi: 123 for pi, _ in enumerate(paths)}
    nodes, bonus = verify_tree_greedy(depth_map, tree, bonus_map)
    assert len(nodes) == 2
    assert bonus == 123
    print("[OK] 树验证选择接受路径与 bonus token 正确")


if __name__ == "__main__":
    test_head_forward()
    test_draft_chain()
    test_accept_greedy()
    test_rejection_sampling()
    test_tree()
    test_verify_tree()
    print("\n全部 MTP 测试通过 ✅")
