from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class SparseIndex:
    """
    Unified container for sparse attention indices.

    Different methods populate different fields:
      - token_mask    : [B, H, Q, K] bool mask (Quest-style token sparsity)
      - block_indices : [B, H, N_active_blocks] or similar (block-sparse)
      - page_indices  : [B, H, N_active_pages] (page-based schemes)
      - extra         : method-specific structured data
    """

    token_mask: Optional[torch.Tensor] = None
    block_indices: Optional[torch.Tensor] = None
    page_indices: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Pattern:
    """
    Pattern generator interface.

    Implementations analyse (q, k, v) and optional metadata, then
    return a method-specific description of the sparse pattern.
    The returned object is intentionally opaque to keep the base
    interface light – IndexBuilder is responsible for turning it
    into a concrete SparseIndex.
    """

    def generate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        stage: str,
        config: Dict[str, Any],
    ) -> Any:
        raise NotImplementedError


class IndexBuilder:
    """
    Converts pattern outputs into a concrete SparseIndex object that
    downstream kernels can understand.
    """

    def build(
        self,
        pattern_output: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        stage: str,
        config: Dict[str, Any],
    ) -> SparseIndex:
        raise NotImplementedError


class KernelExecutor:
    """
    Executes the attention kernel under a given sparse pattern.

    Implementations are free to ignore SparseIndex when they wrap an
    existing monolithic implementation (e.g., current MInference prefill),
    but the interface allows more fine-grained kernels to be added later.
    """

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index: SparseIndex,
        stage: str,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError


class PatternObserver:
    """
    Optional observer interface that can be used to collect statistics
    about the sparse pattern and kernel behaviour without changing the
    core pipeline logic.

    Typical use cases include measuring attention recall, sparsity
    distributions, or stability across steps/layers. Implementations
    should avoid mutating q/k/v/index to keep behaviour unchanged.
    """

    def on_index_built(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index: SparseIndex,
        stage: str,
        config: Dict[str, Any],
    ) -> None:
        return None

    def on_kernel_run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index: SparseIndex,
        stage: str,
        config: Dict[str, Any],
        output: torch.Tensor,
    ) -> None:
        return None


class NoOpPattern(Pattern):
    """
    Default pattern generator that does not impose any sparsity.

    This is used when wrapping dense or already self-contained sparse
    implementations. It simply returns None.
    """

    def generate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        stage: str,
        config: Dict[str, Any],
    ) -> None:
        return None


class NoOpIndexBuilder(IndexBuilder):
    """
    Index builder that always returns an empty SparseIndex.
    """

    def build(
        self,
        pattern_output: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        stage: str,
        config: Dict[str, Any],
    ) -> SparseIndex:
        return SparseIndex()


class PassthroughKernel(KernelExecutor):
    """
    Simple kernel executor that calls a user-provided callable
    with signature (q, k, v, stage, config) -> Tensor.

    This is primarily used to wrap existing implementations like
    minference_prefill_forward, flexprefill_forward, etc., in the
    SAttnF pipeline without rewriting their internals.
    """

    def __init__(self, fn):
        self.fn = fn

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index: SparseIndex,
        stage: str,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        return self.fn(q, k, v, stage, config)
