from __future__ import annotations

from typing import Any, Dict

import torch

from ..modules.minference_forward import (
    compute_vertical_and_slash_indices,
    flash_attn_func,
)
from ..ops.block_sparse_flash_attention import (
    block_sparse_attention_from_index,
    build_block_index,
)
from ..ops.pit_sparse_flash_attention_v2 import (
    build_vertical_slash_index,
    vertical_slash_sparse_attention_from_index,
)
from ..ops.streaming_kernel import streaming_forward
from .base import IndexBuilder, KernelExecutor, Pattern, SparseIndex


class MinferencePattern(Pattern):
    """
    Pattern generator for MInference.

    For now, this class does not recompute any statistics from (q, k, v).
    Instead, it simply exposes the offline `best_pattern` configuration
    for the current layer as a structured object so that SAttnF can
    reason about it uniformly with other methods.

    This keeps the original MInference behaviour unchanged while still
    fitting into the Pattern/Index/Kernel abstraction.
    """

    def generate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        stage: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        attn_cfg = config.get("attn_forward_config", {}) or {}
        best_pattern = attn_cfg.get("best_pattern")
        layer_idx = config.get("layer_idx", 0)

        # Extract the per-layer head pattern if available. The JSON is
        # typically a list of length `num_hidden_layers`, each entry a
        # dict: head_id(str) -> [ty, vertical_size, slash_size, score].
        layer_pattern = None
        if isinstance(best_pattern, list) and 0 <= layer_idx < len(best_pattern):
            layer_pattern = best_pattern[layer_idx]

        return {
            "layer_idx": layer_idx,
            "layer_pattern": layer_pattern,
            "starting_layer": attn_cfg.get("starting_layer", 0),
            "is_search": attn_cfg.get("is_search", False),
            "minference_ratio": attn_cfg.get("minference_ratio"),
        }


class MinferenceIndexBuilder(IndexBuilder):
    """
    Index builder for MInference.

    MInference computes its sparse pattern and index internally in
    `minference_prefill_kernel` using the offline configuration and
    QK statistics. To avoid changing that behaviour, we do not build
    an explicit index here. Instead, we store the pattern metadata in
    `SparseIndex.extra` for potential inspection.
    """

    def build(
        self,
        pattern_output: Dict[str, Any],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        stage: str,
        config: Dict[str, Any],
    ) -> SparseIndex:
        """
        Construct all indices needed by the Minference kernels.

        This method mirrors the internal logic of `minference_prefill_kernel`
        but restricts itself to index construction:

          - For `vertical_and_slash` heads, it calls
            `compute_vertical_and_slash_indices` followed by
            `build_vertical_slash_index` to obtain both token-level and
            block-level indices.

          - For `block_sparse` heads, it follows the same padding and
            `_build_block_index` usage as `block_sparse_attention`.

        The resulting indices are stored in `SparseIndex.extra` and
        consumed by `MinferenceKernelExecutor`, so that the executor
        never needs to re-build indices.
        """
        extra: Dict[str, Any] = {"minference_pattern": pattern_output}

        if stage != "prefill":
            # MInference sparsity is currently only defined for prefill.
            return SparseIndex(extra=extra)

        layer_pattern = pattern_output.get("layer_pattern")
        if layer_pattern is None:
            # No offline pattern available (e.g., search not run); we
            # simply return the pattern metadata.
            return SparseIndex(extra=extra)

        bsz, num_heads, q_len, head_dim = q.shape
        minference_ratio = pattern_output.get("minference_ratio")

        # Per-head containers
        v_idx_per_head: Dict[int, torch.Tensor] = {}
        s_idx_per_head: Dict[int, torch.Tensor] = {}
        vs_block_count: Dict[int, torch.Tensor] = {}
        vs_block_offset: Dict[int, torch.Tensor] = {}
        vs_column_count: Dict[int, torch.Tensor] = {}
        vs_column_index: Dict[int, torch.Tensor] = {}
        block_index_per_head: Dict[int, torch.Tensor] = {}

        # Match the original implementation, which uses a single seqlen
        # tensor of length 1 for all batches.
        seqlens = torch.tensor([q_len], dtype=torch.int32, device=q.device)

        for head in range(num_heads):
            spec = layer_pattern.get(str(head)) if layer_pattern is not None else None
            if spec is None:
                # Default used by `minference_prefill_kernel`.
                spec = ("vertical_and_slash", 1000, 6096, 1)
            ty, vertical_size, slash_size, _ = spec

            if minference_ratio is not None:
                vertical_size = int(vertical_size * minference_ratio)
                slash_size = int(slash_size * minference_ratio)

            q_h = q[:, head : head + 1, :, :]
            k_h = k[:, head : head + 1, :, :]

            if ty == "vertical_and_slash":
                # 1) token-level indices (consistent with the native path)
                v_tok, s_tok = compute_vertical_and_slash_indices(
                    q_h, k_h, vertical_size, slash_size
                )
                # compress to [B, 1, NNZ]
                v_tok = v_tok.reshape(bsz, 1, -1)
                s_tok = s_tok.reshape(bsz, 1, -1)
                v_idx_per_head[head] = v_tok
                s_idx_per_head[head] = s_tok

                # 2) block-level indices required by the kernels
                (
                    blk_count,
                    blk_offset,
                    col_count,
                    col_index,
                ) = build_vertical_slash_index(
                    seqlens,
                    v_tok,
                    s_tok,
                    q_len,
                )
                vs_block_count[head] = blk_count
                vs_block_offset[head] = blk_offset
                vs_column_count[head] = col_count
                vs_column_index[head] = col_index

            elif ty == "block_sparse":
                # Follow `block_sparse_attention` to build block indices.
                topk = 100
                block_size_M = 64
                block_size_N = 64
                pad = block_size_M - (q_len & (block_size_M - 1))
                if pad != 0:
                    q_pad = torch.nn.functional.pad(
                        q_h, [0, 0, 0, pad, 0, 0, 0, 0]
                    )
                    k_pad = torch.nn.functional.pad(
                        k_h, [0, 0, 0, pad, 0, 0, 0, 0]
                    )
                else:
                    q_pad = q_h
                    k_pad = k_h

                blk_index = build_block_index(
                    q_pad,
                    k_pad,
                    topk,
                    block_size_M=block_size_N,
                    block_size_N=block_size_N,
                )
                block_index_per_head[head] = blk_index

            else:
                # `stream_llm` and any other fallback types do not require
                # explicit indices beyond what the kernel computes internally.
                continue

        if v_idx_per_head:
            extra["v_idx"] = v_idx_per_head
            extra["s_idx"] = s_idx_per_head
            extra["vs_block_count"] = vs_block_count
            extra["vs_block_offset"] = vs_block_offset
            extra["vs_column_count"] = vs_column_count
            extra["vs_column_index"] = vs_column_index
        if block_index_per_head:
            extra["block_index"] = block_index_per_head

        return SparseIndex(extra=extra)


class MinferenceKernelExecutor(KernelExecutor):
    """
    Kernel executor for MInference.

    To preserve the original behaviour exactly, we delegate the actual
    computation to `minference_prefill_forward` for the pre-filling
    stage. This ensures that any future changes in the original
    implementation are automatically reflected here.
    In this SAttnF-specific executor, we re-implement the per-head
    control flow of `minference_prefill_forward` in a lightweight way,
    but still call the original `minference_prefill_kernel` for the
    actual attention computation. This gives us a clearer separation
    between the pattern layer (offline `best_pattern`) and the kernel
    layer, while keeping the numerics identical to the original
    MInference implementation.
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
        if stage != "prefill":
            raise NotImplementedError(
                "MinferenceKernelExecutor currently only supports the prefill stage."
            )

        attn_cfg = config.get("attn_forward_config", {}) or {}
        layer_idx: int = config.get("layer_idx", 0)

        # Pattern metadata computed by MinferencePattern.
        pattern_meta: Dict[str, Any] = index.extra.get("minference_pattern", {}) or {}
        layer_pattern = pattern_meta.get("layer_pattern")
        starting_layer: int = pattern_meta.get(
            "starting_layer", attn_cfg.get("starting_layer", 0)
        )
        is_search: bool = pattern_meta.get("is_search", attn_cfg.get("is_search", False))
        minference_ratio = pattern_meta.get("minference_ratio")

        if is_search:
            # The SAttnF path is intended for inference with an existing
            # offline pattern. Training-time search remains handled by
            # the original `minference_prefill_forward`.
            raise NotImplementedError(
                "Search mode is not supported via SAttnF; please use "
                "attn_type='minference' for pattern search."
            )

        bsz, num_heads, q_len, head_dim = q.shape
        out = torch.empty_like(q)

        # Indices prepared by MinferenceIndexBuilder.
        v_idx_dict: Dict[int, torch.Tensor] = index.extra.get("v_idx", {})
        s_idx_dict: Dict[int, torch.Tensor] = index.extra.get("s_idx", {})
        vs_block_count: Dict[int, torch.Tensor] = index.extra.get(
            "vs_block_count", {}
        )
        vs_block_offset: Dict[int, torch.Tensor] = index.extra.get(
            "vs_block_offset", {}
        )
        vs_column_count: Dict[int, torch.Tensor] = index.extra.get(
            "vs_column_count", {}
        )
        vs_column_index: Dict[int, torch.Tensor] = index.extra.get(
            "vs_column_index", {}
        )
        block_index_dict: Dict[int, torch.Tensor] = index.extra.get(
            "block_index", {}
        )

        for head in range(num_heads):
            q_h = q[:, head : head + 1, :, :]
            k_h = k[:, head : head + 1, :, :]
            v_h = v[:, head : head + 1, :, :]

            if layer_idx >= starting_layer:
                # Decode the per-head pattern. Fall back to the same
                # default as `minference_prefill_kernel` if missing.
                spec = layer_pattern.get(str(head)) if layer_pattern is not None else None
                if spec is None:
                    spec = ("vertical_and_slash", 1000, 6096, 1)
                ty, vertical_size, slash_size, _ = spec
                if minference_ratio is not None:
                    vertical_size = int(vertical_size * minference_ratio)
                    slash_size = int(slash_size * minference_ratio)

                if ty == "stream_llm":
                    # Uses the shared streaming kernel directly; no explicit
                    # sparse indices are required.
                    attn_out = streaming_forward(
                        q_h,
                        k_h,
                        v_h,
                        vertical_size,
                        slash_size,
                    )

                elif ty == "vertical_and_slash":
                    # Consume pre-built block indices from the index
                    # builder. If anything is missing, fall back to the
                    # original kernel to avoid behaviour changes.
                    blk_count = vs_block_count.get(head)
                    blk_offset = vs_block_offset.get(head)
                    col_count = vs_column_count.get(head)
                    col_index = vs_column_index.get(head)

                    if (
                        blk_count is None
                        or blk_offset is None
                        or col_count is None
                        or col_index is None
                    ):
                        # Safety fallback: use the monolithic kernel.
                        from ..modules.minference_forward import (
                            minference_prefill_kernel as _native_kernel,
                        )

                        attn_out = _native_kernel(
                            q_h,
                            k_h,
                            v_h,
                            head,
                            layer_idx,
                            attn_cfg,
                        )
                    else:
                        # Replicate the padding and head-dim alignment
                        # logic from `vertical_slash_sparse_attention`,
                        # but reuse the pre-computed block indices.
                        block_size_M = 64
                        block_size_N = 64
                        context_size = q_len
                        pad = (block_size_M - context_size) & (block_size_M - 1)
                        if pad != 0:
                            q_pad = torch.nn.functional.pad(
                                q_h, [0, 0, 0, pad, 0, 0, 0, 0]
                            )
                            k_pad = torch.nn.functional.pad(
                                k_h, [0, 0, 0, pad, 0, 0, 0, 0]
                            )
                            v_pad = torch.nn.functional.pad(
                                v_h, [0, 0, 0, pad, 0, 0, 0, 0]
                            )
                        else:
                            q_pad, k_pad, v_pad = q_h, k_h, v_h

                        head_dim_eff = head_dim
                        if head_dim_eff not in (16, 32, 64, 128, 256, 512):
                            target_dim = 2 ** int(torch.ceil(torch.log2(torch.tensor(head_dim_eff)))) - head_dim_eff
                            q_pad = torch.nn.functional.pad(
                                q_pad, [0, target_dim, 0, 0, 0, 0, 0, 0]
                            )
                            k_pad = torch.nn.functional.pad(
                                k_pad, [0, target_dim, 0, 0, 0, 0, 0, 0]
                            )
                            v_pad = torch.nn.functional.pad(
                                v_pad, [0, target_dim, 0, 0, 0, 0, 0, 0]
                            )

                        seqlens = torch.tensor(
                            [context_size],
                            dtype=torch.int32,
                            device=q_h.device,
                        )
                        attn_out = vertical_slash_sparse_attention_from_index(
                            q_pad,
                            k_pad,
                            v_pad,
                            seqlens,
                            blk_count,
                            blk_offset,
                            col_count,
                            col_index,
                            context_size=context_size,
                            head_dim=head_dim,
                            block_size_M=block_size_M,
                            block_size_N=block_size_N,
                        )

                elif ty == "block_sparse":
                    blk_index = block_index_dict.get(head)
                    if blk_index is None:
                        from ..modules.minference_forward import (
                            minference_prefill_kernel as _native_kernel,
                        )

                        attn_out = _native_kernel(
                            q_h,
                            k_h,
                            v_h,
                            head,
                            layer_idx,
                            attn_cfg,
                        )
                    else:
                        block_size_M = 64
                        block_size_N = 64
                        context_size = q_len
                        pad = block_size_M - (context_size & (block_size_M - 1))
                        if pad != 0:
                            q_pad = torch.nn.functional.pad(
                                q_h, [0, 0, 0, pad, 0, 0, 0, 0]
                            )
                            k_pad = torch.nn.functional.pad(
                                k_h, [0, 0, 0, pad, 0, 0, 0, 0]
                            )
                            v_pad = torch.nn.functional.pad(
                                v_h, [0, 0, 0, pad, 0, 0, 0, 0]
                            )
                        else:
                            q_pad, k_pad, v_pad = q_h, k_h, v_h

                        seqlens = torch.tensor(
                            [context_size],
                            dtype=torch.int32,
                            device=q_h.device,
                        )
                        attn_out = block_sparse_attention_from_index(
                            q_pad,
                            k_pad,
                            v_pad,
                            seqlens,
                            blk_index,
                            context_size=context_size,
                            block_size_M=block_size_M,
                            block_size_N=block_size_N,
                        )

            else:
                q_mat = q_h.transpose(1, 2)  # [B, L, 1, D]
                k_mat = k_h.transpose(1, 2)
                v_mat = v_h.transpose(1, 2)
                attn_out = flash_attn_func(
                    q_mat,
                    k_mat,
                    v_mat,
                    0.0,
                    softmax_scale=None,
                    causal=q_len != 1,
                ).view(bsz, 1, q_len, head_dim)

            out[:, head : head + 1, :, :] = attn_out

        return out
