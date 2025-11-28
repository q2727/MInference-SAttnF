from __future__ import annotations

from typing import Any, Dict

import torch

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.distributed import get_tensor_model_parallel_rank

from .router import sattnf_prefill_forward


def sattnf_vllm_forward(
    pattern_config,
    vllm_version: str = "0.4.1",
    patch_config: Dict[str, Any] | None = None,
):
    """
    Generic vLLM attention forward that routes the prefill stage through
    the SAttnF dispatcher while keeping the original PagedAttention
    decode path intact.

    This is the vLLM analogue of `sattnf_prefill_forward` used on the
    HF path. The concrete sparse-attention method is selected via:

        attn_forward_config["sattnf_method"]

    which defaults to "minference".
    """
    if patch_config is None:
        patch_config = {}

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        kv_scale: float,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            query: [num_tokens, num_heads * head_dim]
            key:   [num_tokens, num_kv_heads * head_dim]
            value: [num_tokens, num_kv_heads * head_dim]
        Returns:
            [num_tokens, num_heads * head_dim]
        """
        num_tokens, hidden_size = query.shape

        # Reshape QKV to vLLM's internal representation.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                attn_metadata.kv_cache_dtype,
                kv_scale,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)

        # Split prefill / decode.
        decode_query = query[num_prefill_tokens:]
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        # ---------------------- Prefill via SAttnF ----------------------
        if prefill_meta := attn_metadata.prefill_metadata:
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # For now we only support the simple "no prefix" case in the
                # SAttnF path, consistent with the original Minference-vLLM
                # integration.

                # vLLM packs multiple sequences; here we assume a single
                # contiguous prompt and treat it as batch=1.
                # [seq_len, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]
                q = query.unsqueeze(0).transpose(1, 2)
                k = key.unsqueeze(0).transpose(1, 2)
                v = value.unsqueeze(0).transpose(1, 2)

                attn_forward_config = {
                    # Provide the offline pattern config (e.g. Minference).
                    "best_pattern": pattern_config,
                    **patch_config,
                }
                prefill_kwargs = {
                    "attention_mask": None,
                    "layer_idx": layer_idx,
                    "num_hidden_layers": len(pattern_config),
                    "attn_forward_config": attn_forward_config,
                }

                out = sattnf_prefill_forward(q, k, v, prefill_kwargs)
                # [1, num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim]
                out = out.transpose(1, 2).squeeze(0).contiguous()
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # Prefix-enabled attention: keep the original vLLM behaviour.
                key_cache, value_cache = PagedAttention.split_kv_cache(
                    kv_cache, self.num_kv_heads, self.head_size
                )
                output[:num_prefill_tokens] = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.subquery_start_loc,
                    prefill_meta.prompt_lens_tensor,
                    prefill_meta.context_lens,
                    prefill_meta.max_subquery_len,
                    self.alibi_slopes,
                )

        # ------------------------ Decode (unchanged) ------------------------
        if decode_meta := attn_metadata.decode_metadata:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )
            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.context_lens,
                decode_meta.max_context_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        return output.view(num_tokens, hidden_size)

    return forward

