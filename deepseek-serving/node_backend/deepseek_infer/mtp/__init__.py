"""MTP 多 token 预测与投机解码。"""
from .mtp_head import MTPHead, run_block
from .speculative import (Tree, accept_greedy, draft_chain, draft_tree,
                          rejection_accept_one, verify_tree_greedy)

__all__ = ["MTPHead", "run_block", "Tree", "accept_greedy", "draft_chain", "draft_tree",
           "rejection_accept_one", "verify_tree_greedy"]
