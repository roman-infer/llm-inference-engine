# -*- coding: utf-8 -*-
"""
固定大小状态缓存（StateCache）。

线性/混合架构没有随上下文增长的 KV，每层只有一个固定大小的状态：
    mamba1/mamba2 : dict（含 S 与 conv 窗口）
    gdn/kda       : Tensor（状态矩阵）
    swa           : tuple（窗口 K/V）
StateCache 按序列 ID 存取这些状态，并支持 fork（前缀分支复用）。

与"分页 KV 池 + CoW"的本质区别：
    状态是紧凑的定长矩阵，复制整张状态的成本很低，
    因此分支复用直接做深拷贝即可，不需要页表、引用计数和写时复制。
    这正是"O(N) KV 变 O(1) state"在缓存管理上的直接收益。
"""
import copy
import torch


class StateCache:
    """多序列状态缓存（内存中）。"""

    def __init__(self):
        self._store = {}

    def put(self, seq_id: str, states: list):
        """保存某序列的状态列表（每层一个）。"""
        self._store[seq_id] = states

    def get(self, seq_id: str):
        """读取状态；不存在返回 None。"""
        return self._store.get(seq_id, None)

    def contains(self, seq_id: str) -> bool:
        return seq_id in self._store

    def fork(self, parent_id: str, child_id: str) -> bool:
        """
        从父序列分叉出子序列（并行采样/树搜索场景）。
        深拷贝父状态作为子序列起点；返回是否成功。
        """
        if parent_id not in self._store:
            return False
        self._store[child_id] = copy.deepcopy(self._store[parent_id])
        return True

    def drop(self, seq_id: str):
        """删除某序列状态，释放缓存。"""
        self._store.pop(seq_id, None)

    def __len__(self):
        return len(self._store)

    def ids(self):
        return list(self._store.keys())


def clone_states(states: list):
    """对一组层状态做整体深拷贝（结构可能是 dict/tensor/tuple）。"""
    return copy.deepcopy(states)


def detach_states(states: list):
    """把状态从计算图分离（生成时避免持有中间激活）。"""
    def _detach(s):
        if torch.is_tensor(s):
            return s.detach()
        if isinstance(s, dict):
            return {k: _detach(v) for k, v in s.items()}
        if isinstance(s, tuple):
            return tuple(_detach(v) for v in s)
        return s
    return [_detach(s) for s in states]
