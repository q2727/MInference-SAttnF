from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch

from ..modules.flexprefill import flexprefill_forward
from ..modules.minference_forward import minference_prefill_forward
from ..modules.xattention import xattention_forward
from .minference_impl import (
    MinferenceIndexBuilder,
    MinferenceKernelExecutor,
    MinferencePattern,
)
from .quest_impl import QuestIndexBuilder, QuestKernelExecutor, QuestPattern
from .base import (
    IndexBuilder,
    KernelExecutor,
    NoOpIndexBuilder,
    NoOpPattern,
    PassthroughKernel,
    Pattern,
    PatternObserver,
    SparseIndex,
)


@dataclass
class SAttnFMethod:
    """
    A concrete sparse-attention method expressed as a triple:
      Pattern -> IndexBuilder -> KernelExecutor

    An optional PatternObserver can be attached to collect statistics
    (e.g., attention recall) without changing the pipeline semantics.
    """

    pattern: Pattern
    index_builder: IndexBuilder
    kernel: KernelExecutor
    observer: Optional[PatternObserver] = None

    def run_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        # Today we only distinguish prefill/decoding at this level.
        stage = "prefill"
        pattern_out = self.pattern.generate(q, k, v, stage, config)
        index = self.index_builder.build(pattern_out, q, k, v, stage, config)
        if self.observer is not None:
            self.observer.on_index_built(q, k, v, index, stage, config)
        out = self.kernel.run(q, k, v, index, stage, config)
        if self.observer is not None:
            self.observer.on_kernel_run(q, k, v, index, stage, config, out)
        return out

    def run_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        stage = "decode"
        pattern_out = self.pattern.generate(q, k, v, stage, config)
        index = self.index_builder.build(pattern_out, q, k, v, stage, config)
        if self.observer is not None:
            self.observer.on_index_built(q, k, v, index, stage, config)
        out = self.kernel.run(q, k, v, index, stage, config)
        if self.observer is not None:
            self.observer.on_kernel_run(q, k, v, index, stage, config, out)
        return out


class SAttnFDispatcher:
    """
    Lightweight dispatcher that maps method names to SAttnFMethod
    instances and drives the 3-stage pipeline:

        Pattern -> IndexBuilder -> KernelExecutor

    The KV/cache handling is performed by the existing cache classes
    before we see the tensors here.
    """

    def __init__(self):
        self._methods: Dict[str, SAttnFMethod] = {}
        self._register_builtin_methods()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run_prefill(
        self,
        method: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        m = self._methods.get(method)
        if m is None:
            raise ValueError(f"SAttnF method '{method}' is not registered.")
        return m.run_prefill(q, k, v, config)

    def run_decode(
        self,
        method: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        m = self._methods.get(method)
        if m is None:
            raise ValueError(f"SAttnF method '{method}' is not registered.")
        return m.run_decode(q, k, v, config)

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register_method(self, name: str, method: SAttnFMethod) -> None:
        if name in self._methods:
            raise ValueError(f"SAttnF method '{name}' already registered.")
        self._methods[name] = method

    def set_observer(self, name: str, observer: Optional[PatternObserver]) -> None:
        """
        Attach or replace a PatternObserver for a registered method.

        This allows users to collect statistics (e.g., attention recall)
        without changing the core pipeline or re-registering methods.
        """
        method = self._methods.get(name)
        if method is None:
            raise ValueError(f"SAttnF method '{name}' is not registered.")
        method.observer = observer

    def _register_builtin_methods(self) -> None:
        """
        Register thin wrappers around existing implementations.

        These use NoOpPattern / NoOpIndexBuilder and simply forward
        to the respective prefill functions.  This provides an initial
        end-to-end pipeline without changing behaviour.
        """

        self.register_method(
            "minference",
            SAttnFMethod(
                pattern=MinferencePattern(),
                index_builder=MinferenceIndexBuilder(),
                kernel=MinferenceKernelExecutor(),
            ),
        )

        # Quest: decode-time sparse attention based on heavy-hitter
        # selection over the cached KV. Prefill remains dense; the
        # SAttnF integration currently only targets the decode stage.
        self.register_method(
            "quest",
            SAttnFMethod(
                pattern=QuestPattern(),
                index_builder=QuestIndexBuilder(),
                kernel=QuestKernelExecutor(),
            ),
        )

        def _flexprefill_kernel(q, k, v, stage: str, cfg: Dict[str, Any]) -> torch.Tensor:
            if stage != "prefill":
                raise NotImplementedError(
                    "flexprefill decode via SAttnF is not implemented yet."
                )
            prefill_cfg = cfg.get("attn_forward_config", {})
            prefill_kwargs = {"attn_forward_config": prefill_cfg}
            return flexprefill_forward(q, k, v, prefill_kwargs)

        self.register_method(
            "flexprefill",
            SAttnFMethod(
                pattern=NoOpPattern(),
                index_builder=NoOpIndexBuilder(),
                kernel=PassthroughKernel(_flexprefill_kernel),
            ),
        )

        def _xattention_kernel(q, k, v, stage: str, cfg: Dict[str, Any]) -> torch.Tensor:
            if stage != "prefill":
                raise NotImplementedError(
                    "xattention decode via SAttnF is not implemented yet."
                )
            prefill_cfg = cfg.get("attn_forward_config", {})
            prefill_kwargs = {"attn_forward_config": prefill_cfg}
            return xattention_forward(q, k, v, prefill_kwargs)

        self.register_method(
            "xattention",
            SAttnFMethod(
                pattern=NoOpPattern(),
                index_builder=NoOpIndexBuilder(),
                kernel=PassthroughKernel(_xattention_kernel),
            ),
        )


_GLOBAL_DISPATCHER: SAttnFDispatcher | None = None


def get_sattnf_dispatcher() -> SAttnFDispatcher:
    """
    Returns a process-local singleton dispatcher.
    """

    global _GLOBAL_DISPATCHER
    if _GLOBAL_DISPATCHER is None:
        _GLOBAL_DISPATCHER = SAttnFDispatcher()
    return _GLOBAL_DISPATCHER


def sattnf_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefill_kwargs: Dict[str, Any],
) -> torch.Tensor:
    """
    Entry point compatible with minference.modules.forward.

    q, k, v: [B, num_heads, q_len, head_dim]
    prefill_kwargs:
        - attention_mask
        - layer_idx
        - num_hidden_layers
        - attn_forward_config
    """

    dispatcher = get_sattnf_dispatcher()
    cfg = {
        "attention_mask": prefill_kwargs.get("attention_mask"),
        "layer_idx": prefill_kwargs.get("layer_idx", 0),
        "num_hidden_layers": prefill_kwargs.get("num_hidden_layers", 0),
        "attn_forward_config": prefill_kwargs.get("attn_forward_config", {}),
    }
    method_name = cfg["attn_forward_config"].get("sattnf_method", "minference")
    return dispatcher.run_prefill(method_name, q, k, v, cfg)


def sattnf_decoding_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decoding_kwargs: Dict[str, Any],
) -> torch.Tensor:
    dispatcher = get_sattnf_dispatcher()
    cfg = {
        "layer_idx": decoding_kwargs.get("layer_idx", 0),
        "attn_forward_config": decoding_kwargs.get("attn_forward_config", {}),
        "attention_mask": decoding_kwargs.get("attention_mask"),
        "position_ids": decoding_kwargs.get("position_ids"),
        "num_key_value_groups": decoding_kwargs.get("num_key_value_groups", 1),
    }
    method_name = cfg["attn_forward_config"].get(
        "decode_sattnf_method",
        cfg["attn_forward_config"].get("sattnf_method", "minference"),
    )
    return dispatcher.run_decode(method_name, q, k, v, cfg)
