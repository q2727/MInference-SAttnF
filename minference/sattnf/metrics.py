from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from .base import PatternObserver, SparseIndex


Stage = Literal["prefill", "decode"]


@dataclass
class AttentionRecallRecord:
    stage: Stage
    layer_idx: int
    mean_recall: float
    per_head_recall: torch.Tensor  # [H]
    num_queries: int
    topk: int


class AttentionRecallObserver(PatternObserver):
    """
    Observer that measures how well a sparse pattern "recalls" the
    highest‑probability entries of the dense attention distribution.

    The metric is defined as:

        Recall@K(b, h, i) = | TopK_dense(b,h,i) ∩ S(b,h,i) | / K

    where TopK_dense(b,h,i) are the indices of the top‑K keys under
    dense attention for query token i, and S(b,h,i) is the set of keys
    selected by the sparse pattern.

    Notes:
      - For now this implementation requires `index.token_mask` to be
        present with shape [B, H, Q, K]. Methods that encode sparsity
        via other index structures are currently ignored.
      - Computation is restricted to at most `max_q` query positions
        per call to keep overhead reasonable.
    """

    def __init__(
        self,
        topk: int = 32,
        max_q: int = 32,
        stages: Tuple[Stage, ...] = ("prefill", "decode"),
    ) -> None:
        self.topk = topk
        self.max_q = max_q
        self.stages = set(stages)
        self.records: List[AttentionRecallRecord] = []

    # ------------------------------------------------------------------
    # PatternObserver API
    # ------------------------------------------------------------------
    def on_index_built(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index: SparseIndex,
        stage: str,
        config: Dict[str, Any],
    ) -> None:
        # Stage filter
        if stage not in self.stages:
            return

        # We currently require an explicit token_mask to compute recall.
        mask = index.token_mask
        if mask is None:
            return

        # Expected shapes:
        #   q    : [B, H, Q, D]
        #   k    : [B, H, K, D]
        #   mask : [B, H, Q, K]
        if q.dim() != 4 or k.dim() != 4 or mask.dim() != 4:
            return

        bsz, num_heads, q_len, head_dim = q.shape
        _, _, kv_len, _ = k.shape
        _, _, mask_q_len, mask_k_len = mask.shape
        if mask_q_len != q_len or mask_k_len != kv_len:
            # Shape mismatch; skip silently to avoid disrupting the run.
            return

        if kv_len == 0 or q_len == 0:
            return

        # Select a suffix of the query positions to bound the cost.
        q_count = min(self.max_q, q_len)
        q_start = q_len - q_count
        q_indices = torch.arange(
            q_start, q_len, device=q.device, dtype=torch.long
        )  # [Q_sub]

        # Slice tensors: [B, H, Q_sub, ...]
        q_sub = q[:, :, q_start:q_len, :]
        mask_sub = mask[:, :, q_start:q_len, :]

        # Flatten batch and head dimensions for easier matmul.
        q_flat = q_sub.reshape(bsz * num_heads, q_count, head_dim)
        k_flat = k.reshape(bsz * num_heads, kv_len, head_dim)
        mask_flat = mask_sub.reshape(bsz * num_heads, q_count, kv_len)

        # Compute dense attention scores and probabilities.
        with torch.no_grad():
            q_f32 = q_flat.to(dtype=torch.float32)
            k_f32 = k_flat.to(dtype=torch.float32)
            scale = 1.0 / math.sqrt(float(head_dim))
            scores = torch.matmul(q_f32, k_f32.transpose(1, 2)) * scale
            dense_attn = torch.softmax(scores, dim=-1)  # [B*H, Q_sub, K]

            # Top‑K indices under dense attention.
            k_topk = min(self.topk, kv_len)
            _, topk_idx = torch.topk(
                dense_attn, k=k_topk, dim=-1, largest=True, sorted=False
            )  # [B*H, Q_sub, K_top]

            # Mask lookup for those indices.
            # mask_flat is bool; gather preserves dtype.
            selected = mask_flat.gather(dim=-1, index=topk_idx)  # [B*H, Q_sub, K_top]
            # Recall per query token.
            recall_per_query = selected.to(dtype=torch.float32).mean(
                dim=-1
            )  # [B*H, Q_sub]

            # Aggregate per head: average over batch and query positions.
            recall_per_query = recall_per_query.view(bsz, num_heads, q_count)
            per_head_recall = recall_per_query.mean(dim=(0, 2))  # [H]
            mean_recall = float(per_head_recall.mean().item())

        layer_idx = int(config.get("layer_idx", 0))
        self.records.append(
            AttentionRecallRecord(
                stage=stage,  # type: ignore[arg-type]
                layer_idx=layer_idx,
                mean_recall=mean_recall,
                per_head_recall=per_head_recall.detach().cpu(),
                num_queries=q_count,
                topk=k_topk,
            )
        )

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
        # This observer only needs the indices and dense scores, so the
        # kernel output is not used for now.
        return None

