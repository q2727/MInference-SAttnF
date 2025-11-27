from __future__ import annotations

from typing import Any, Dict

import torch

from ..modules.quest import build_quest_decode_mask
from .base import IndexBuilder, KernelExecutor, Pattern, SparseIndex


class QuestPattern(Pattern):
    """
    Pattern generator for Quest.

    Quest's sparsity is defined implicitly by its heavy-hitter selection
    over the cached KV. At decode time the main degrees of freedom are
    the `chunk_size` and `token_budget` hyperparameters, which we expose
    via the generic attn_forward_config dictionary.
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
        return {
            "chunk_size": attn_cfg.get("chunk_size", 16),
            "token_budget": attn_cfg.get("token_budget", 1024),
        }


class QuestIndexBuilder(IndexBuilder):
    """
    For Quest we currently keep the index-building logic inside the
    original implementation (see `quest_decode_kernel`). The SAttnF
    index builder simply forwards the hyperparameters so that the
    kernel executor can reconstruct the decoding_kwargs structure.
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
        extra: Dict[str, Any] = {"quest_config": pattern_output}

        if stage != "decode":
            # Quest sparsity is only defined for the decode stage; prefill
            # continues to use the dense FlashAttention path.
            return SparseIndex(extra=extra)

        attn_cfg = config.get("attn_forward_config", {}) or {}
        chunk_size = pattern_output.get("chunk_size", attn_cfg.get("chunk_size", 16))
        token_budget = pattern_output.get(
            "token_budget", attn_cfg.get("token_budget", 1024)
        )

        attention_mask = config.get("attention_mask")
        position_ids = config.get("position_ids")

        # Build the Quest mask using the shared helper. This mirrors the
        # heavy-hitter selection used in `quest_decode_kernel`, so that
        # both the native path and SAttnF see exactly the same pattern.
        mask_bottom = build_quest_decode_mask(
            q,
            k,
            attention_mask,
            position_ids,
            chunk_size,
            token_budget,
        )

        return SparseIndex(token_mask=mask_bottom, extra=extra)


class QuestKernelExecutor(KernelExecutor):
    """
    Kernel executor that delegates to the existing Quest decode kernel.

    At prefill time Quest falls back to dense FlashAttention in the
    original implementation, so here we only support the decode stage
    via SAttnF. Prefill should continue to use the standard dense path.
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
        if stage != "decode":
            raise NotImplementedError(
                "QuestKernelExecutor currently only supports the decode stage "
                "via SAttnF. Prefill should use the existing dense path."
            )

        # Retrieve the pre-built mask from the index. If it is missing
        # for some reason, fall back to the original kernel to avoid
        # behaviour changes.
        mask_bottom = index.token_mask
        if mask_bottom is None:
            from ..modules.quest import quest_decode_kernel as _native_decode

            quest_cfg: Dict[str, Any] = index.extra.get("quest_config", {}) or {}
            decoding_kwargs: Dict[str, Any] = {
                "attn_forward_config": {
                    **config.get("attn_forward_config", {}),
                    **quest_cfg,
                },
                "attention_mask": config.get("attention_mask"),
                "position_ids": config.get("position_ids"),
            }
            return _native_decode(q, k, v, decoding_kwargs)

        attention_mask = config.get("attention_mask")
        kv_seq_len = k.size(-2)
        bsz, _, q_len, _ = q.shape

        # Dense attention score computation followed by application of
        # the pre-built Quest mask.
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / torch.sqrt(
            torch.tensor(q.size(-1), device=q.device, dtype=torch.float32)
        )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(
                    torch.finfo(attn_weights.dtype).min, device=attn_weights.device
                ),
            )

        attn_weights[~mask_bottom] = torch.finfo(attn_weights.dtype).min

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output
