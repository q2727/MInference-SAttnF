"""
Lightweight SAttnF (Sparse Attention Framework) scaffolding.

This module defines a small set of abstractions that wrap existing
MInference sparse-attention implementations behind a common pipeline:

    1) (optional) KV / cache handling (performed today by cache classes)
    2) pattern generation            (Pattern)
    3) sparse index construction     (IndexBuilder)
    4) kernel execution              (KernelExecutor)

The initial implementation keeps these layers very thin and forwards
most of the heavy lifting to the existing modules (minference_forward,
flexprefill, xattention, quest, retr_attn, etc.).  The goal is to
centralise control-flow while avoiding behaviour changes.

Over time, additional Pattern / IndexBuilder / KernelExecutor
implementations can be added here without touching the high-level
attention forwarding logic in minference.modules.forward.
"""

from .base import SparseIndex, Pattern, IndexBuilder, KernelExecutor
from .router import SAttnFDispatcher, get_sattnf_dispatcher

__all__ = [
    "SparseIndex",
    "Pattern",
    "IndexBuilder",
    "KernelExecutor",
    "SAttnFDispatcher",
    "get_sattnf_dispatcher",
]

