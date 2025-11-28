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


@dataclass
class DistanceDistributionRecord:
    stage: Stage
    layer_idx: int
    mean_distance: float
    hist: torch.Tensor  # [num_bins], probability over distance buckets
    bin_size: int
    max_distance: int
    num_samples: int


class DistanceDistributionObserver(PatternObserver):
    """
    Observer that measures the distribution of attention distances
    implied by a sparse token mask.

    For each selected entry S[b, h, i, j] == True, we define the
    (causal) distance as:

        d = i - j

    where i is the query position and j the key position. We then
    accumulate a histogram over d for the last `max_q` query tokens.

    Notes:
      - Requires `index.token_mask` with shape [B, H, Q, K].
      - We bucket distances into uniform bins of width `bin_size`
        up to `max_distance`; any larger distance falls into the
        last bucket.
    """

    def __init__(
        self,
        max_q: int = 32,
        bin_size: int = 64,
        max_distance: int = 2048,
        stages: Tuple[Stage, ...] = ("prefill", "decode"),
    ) -> None:
        self.max_q = max_q
        self.bin_size = bin_size
        self.max_distance = max_distance
        self.stages = set(stages)
        self.records: List[DistanceDistributionRecord] = []

        # Number of histogram buckets: [0, bin_size), [bin_size, 2*bin_size), ...,
        # [max_distance, +inf)
        self.num_bins = max_distance // bin_size + 1

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

        # Only support methods that expose an explicit token_mask
        # (e.g., Quest). Minference and other schemes that encode
        # sparsity via custom index structures are intentionally
        # ignored here.
        mask = index.token_mask
        if mask is None or mask.dim() != 4:
            return

        bsz, num_heads, q_len, kv_len = mask.shape
        if q_len == 0 or kv_len == 0:
            return

        # Restrict to the last max_q query positions to limit cost.
        q_count = min(self.max_q, q_len)
        q_start = q_len - q_count
        mask_sub = mask[:, :, q_start:q_len, :]  # [B, H, Q_sub, K]

        # Build absolute positions for queries and keys.
        device = mask_sub.device
        # Approximate absolute query positions. For prefill we have
        # kv_len == q_len, so this reduces to [0, ..., q_len-1]. During
        # decode, q_len is typically 1 and kv_len > q_len; the absolute
        # positions become [kv_len - q_len, ..., kv_len - 1], i.e., the
        # last q_len tokens in the sequence.
        base = kv_len - q_len
        q_local = torch.arange(
            q_start, q_len, device=device, dtype=torch.long
        )  # [Q_sub] in [q_start, q_len)
        q_abs = base + q_local  # [Q_sub]
        k_pos = torch.arange(kv_len, device=device, dtype=torch.long)  # [K]

        # Broadcast positions over batch and heads so that `dist` has
        # the same shape as `mask_sub`: [B, H, Q_sub, K].
        q_pos = q_abs.view(1, 1, q_count, 1).expand(bsz, num_heads, q_count, 1)
        k_pos = k_pos.view(1, 1, 1, kv_len).expand(bsz, num_heads, 1, kv_len)

        # Causal distance d = i - j (clamped to >= 0 to be safe).
        dist = (q_pos - k_pos).clamp(min=0)  # [B, H, Q_sub, K]

        # Select distances where the sparse mask is active.
        selected_dist = dist[mask_sub]  # [N_active]
        if selected_dist.numel() == 0:
            return

        selected_dist = selected_dist.to(dtype=torch.float32)

        # Bucket distances into uniform bins; last bin collects all
        # distances >= max_distance.
        bin_idx = (selected_dist / float(self.bin_size)).floor().to(torch.long)
        bin_idx = torch.clamp(bin_idx, min=0, max=self.num_bins - 1)
        hist = torch.bincount(bin_idx, minlength=self.num_bins).to(torch.float32)

        num_samples = int(hist.sum().item())
        if num_samples == 0:
            return

        hist_prob = (hist / hist.sum()).cpu()
        mean_distance = float(selected_dist.mean().item())

        layer_idx = int(config.get("layer_idx", 0))
        self.records.append(
            DistanceDistributionRecord(
                stage=stage,  # type: ignore[arg-type]
                layer_idx=layer_idx,
                mean_distance=mean_distance,
                hist=hist_prob,
                bin_size=self.bin_size,
                max_distance=self.max_distance,
                num_samples=num_samples,
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
        # Not used for this metric.
        return None


@dataclass
class TimeStabilityRecord:
    stage: Stage
    layer_idx: int
    step_index: int
    mean_jaccard: float
    per_head_jaccard: torch.Tensor  # [H]


class TimeStabilityObserver(PatternObserver):
    """
    Measures temporal stability of a sparse pattern within a layer by
    comparing the selection sets between consecutive decode steps.

    For each layer ℓ and head h, we consider the set of selected keys
    at step t-1 and t:

        S_{t-1}(h) = { j | mask_{t-1}(b=0,h,i*,j) = 1 }
        S_t(h)     = { j | mask_t(b=0,h,i*,j) = 1 }

    and compute the Jaccard similarity:

        J(h) = |S_{t-1}(h) ∩ S_t(h)| / |S_{t-1}(h) ∪ S_t(h)|

    The per-layer stability is the mean of J(h) over all heads.

    Notes:
      - Only supports methods with token_mask (e.g., Quest) and is
        primarily intended for the decode stage where q_len is small.
      - For simplicity we always use the last query position and the
        first batch element.
    """

    def __init__(
        self,
        stages: Tuple[Stage, ...] = ("decode",),
    ) -> None:
        self.stages = set(stages)
        self.records: List[TimeStabilityRecord] = []
        # Previous masks per layer: layer_idx -> [H, K] bool
        self._prev_masks: Dict[int, torch.Tensor] = {}
        # Step index per layer (monotonic counter).
        self._step_index: Dict[int, int] = {}

    def on_index_built(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index: SparseIndex,
        stage: str,
        config: Dict[str, Any],
    ) -> None:
        if stage not in self.stages:
            return

        mask = index.token_mask
        if mask is None or mask.dim() != 4:
            return

        bsz, num_heads, q_len, kv_len = mask.shape
        if bsz == 0 or q_len == 0 or kv_len == 0:
            return

        # Use the last query position of the first batch element.
        curr = mask[0, :, -1, :]  # [H, K_t]
        layer_idx = int(config.get("layer_idx", 0))

        prev = self._prev_masks.get(layer_idx)
        if prev is None:
            self._prev_masks[layer_idx] = curr.clone()
            self._step_index[layer_idx] = 0
            return

        prev_bool = prev
        curr_bool = curr

        # Align along the key dimension in case kv_len has grown
        # between decode steps (typical for autoregressive decoding).
        K_prev = prev_bool.size(-1)
        K_curr = curr_bool.size(-1)
        K = min(K_prev, K_curr)
        prev_bool = prev_bool[..., :K]
        curr_bool = curr_bool[..., :K]
        inter = (prev_bool & curr_bool).sum(dim=-1)
        union = (prev_bool | curr_bool).sum(dim=-1)
        valid = union > 0
        if not valid.any():
            # No meaningful overlap; just update the cache.
            self._prev_masks[layer_idx] = curr_bool.clone()
            self._step_index[layer_idx] += 1
            return

        jaccard = inter[valid].float() / union[valid].float()  # [H_valid]
        mean_jaccard = float(jaccard.mean().item())

        per_head = torch.zeros(
            num_heads, dtype=torch.float32, device=jaccard.device
        )
        per_head[valid] = jaccard

        step_idx = self._step_index.get(layer_idx, 0) + 1
        self._step_index[layer_idx] = step_idx

        self.records.append(
            TimeStabilityRecord(
                stage=stage,  # type: ignore[arg-type]
                layer_idx=layer_idx,
                step_index=step_idx,
                mean_jaccard=mean_jaccard,
                per_head_jaccard=per_head.cpu(),
            )
        )

        # Update cache for the next step.
        self._prev_masks[layer_idx] = curr_bool.clone()

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


@dataclass
class LayerStabilityRecord:
    stage: Stage
    layer_idx: int
    step_index: int
    mean_jaccard: float
    per_head_jaccard: torch.Tensor  # [H]


class LayerStabilityObserver(PatternObserver):
    """
    Measures cross-layer stability of a sparse pattern by comparing
    the selection sets of adjacent layers within the same decode step.

    For a given decode step t and layers ℓ-1 and ℓ, we define:

        S_{ℓ-1}(h) = { j | Mask_{ℓ-1}(b,h,i*,j) = 1 }
        S_ℓ(h)     = { j | Mask_ℓ(b,h,i*,j)     = 1 }

    and compute the Jaccard similarity:

        J(h) = |S_{ℓ-1}(h) ∩ S_ℓ(h)| / |S_{ℓ-1}(h) ∪ S_ℓ(h)|

    The stability for layer ℓ at step t is the mean of J(h) over heads.

    Notes:
      - Only supports methods with token_mask (e.g., Quest) and is
        primarily intended for the decode stage.
      - We detect decode steps by tracking when the layer_idx sequence
        wraps around (i.e., when it decreases compared to the previous
        call), and reset per-step state accordingly.
    """

    def __init__(
        self,
        stages: Tuple[Stage, ...] = ("decode",),
    ) -> None:
        self.stages = set(stages)
        self.records: List[LayerStabilityRecord] = []

        # Per-step masks: layer_idx -> [H, K] bool
        self._current_masks: Dict[int, torch.Tensor] = {}
        self._last_layer_idx_seen: Optional[int] = None
        self._step_index: int = 0

    def on_index_built(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index: SparseIndex,
        stage: str,
        config: Dict[str, Any],
    ) -> None:
        if stage not in self.stages:
            return

        mask = index.token_mask
        if mask is None or mask.dim() != 4:
            return

        bsz, num_heads, q_len, kv_len = mask.shape
        if bsz == 0 or q_len == 0 or kv_len == 0:
            return

        layer_idx = int(config.get("layer_idx", 0))

        # Detect a new decode step by a wrap-around in layer indices.
        if self._last_layer_idx_seen is not None and layer_idx < self._last_layer_idx_seen:
            self._current_masks.clear()
            self._step_index += 1

        self._last_layer_idx_seen = layer_idx

        # Use the last query position of the first batch element.
        curr = mask[0, :, -1, :]  # [H, K]

        # Compare with the previous layer in the same step, if available.
        prev = self._current_masks.get(layer_idx - 1)
        if prev is not None:
            prev_bool = prev
            curr_bool = curr
            inter = (prev_bool & curr_bool).sum(dim=-1)
            union = (prev_bool | curr_bool).sum(dim=-1)
            valid = union > 0
            if valid.any():
                jaccard = inter[valid].float() / union[valid].float()
                mean_jaccard = float(jaccard.mean().item())

                per_head = torch.zeros(
                    num_heads, dtype=torch.float32, device=jaccard.device
                )
                per_head[valid] = jaccard

                self.records.append(
                    LayerStabilityRecord(
                        stage=stage,  # type: ignore[arg-type]
                        layer_idx=layer_idx,
                        step_index=self._step_index,
                        mean_jaccard=mean_jaccard,
                        per_head_jaccard=per_head.cpu(),
                    )
                )

        # Cache current layer mask for subsequent layers in this step.
        self._current_masks[layer_idx] = curr.clone()

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
