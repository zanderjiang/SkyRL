"""Give Megatron's ``GatedDeltaNet`` an inference path when its GPTModel runs inside vLLM.

WHY. ``zerokl/gptmodel_vllm.py`` runs Megatron's ``GPTModel`` as a vLLM model so that the rollout and
the trainer execute the same code. For softmax layers ``swap_core_attention`` swaps
``SelfAttention.core_attention`` for vLLM's paged ``Attention``, which registers itself with vLLM and
gets a KV cache. Qwen3.5's *other* three-quarters of layers are ``GatedDeltaNet``, and nothing
equivalent existed:

  * ``GatedDeltaNet.forward`` raises ``NotImplementedError`` if handed an ``inference_context``, and
    the wrapper passes none -- so it silently takes its TRAINING branch. On a decode batch of N
    single-token requests that runs one chunk kernel over N tokens from a zero state, treating the
    whole batch as one contiguous sequence. Wrong, and quiet about it.
  * Megatron registers no ``MambaSpec``, so vLLM allocates no GDN state slots and builds no
    ``GDNAttentionMetadata``. There is no slot id and no prefill/decode split to drive decode with.

WHAT THIS DOES. Two pieces, mirroring the softmax swap:

  1. :class:`ZeroKLGDNStateLayer` -- a ``MambaBase`` that exists only to be *registered*. vLLM's KV
     cache manager sees it, reserves mamba state blocks, and hands us a ``GDNAttentionMetadata`` per
     forward (slot ids, prefill/decode split). Its ``kv_cache`` tensors are allocated by vLLM and
     deliberately unused: chunk-consistent decode keeps the boundary state, the open-chunk buffer and
     the conv state inside :class:`~.gdn_chunk_consistent.ChunkConsistentGDN`, because
     ``ssm_state[slot]`` under this scheme means "state at the last chunk boundary" and nothing
     outside our code may interpret it.

  2. :func:`swap_gdn_core` -- rebinds ``GatedDeltaNet.forward`` on each hybrid layer to a version
     that keeps every Megatron module (``in_proj``, ``out_norm``, ``out_proj``, ``A_log``,
     ``dt_bias``, ``conv1d``) and routes only the conv+chunk core through ``ChunkConsistentGDN``.
     The trainer runs those same Megatron modules over ``gdn_ops`` (via ``gdn_fla_shim``), so the two
     runtimes differ in nothing but batching -- which is exactly the invariant zero-KL needs.

Gated on ``SKYRL_ZEROKL_GDN=1``. Engine-side only; the trainer never imports this.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _is_gdn(module) -> bool:
    return type(module).__name__ == "GatedDeltaNet"


_state_cls = None


def state_layer_cls():
    """Build ``ZeroKLGDNStateLayer`` on first use.

    It has to genuinely subclass ``MambaBase``: vLLM's model runner enumerates KV-cache layers with
    ``get_layers_from_vllm_config(config, AttentionLayerBase)``, an ``isinstance`` filter. Duck typing
    would leave the layer invisible, vLLM would allocate no state, and the engine would run with an
    empty ``GDNAttentionMetadata``. The class is built lazily so this module stays importable in the
    trainer process, which has no vLLM.
    """
    global _state_cls

    if _state_cls is not None:
        return _state_cls

    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    class ZeroKLGDNStateLayer(nn.Module, MambaBase):
        """vLLM-visible GDN state layer: owns the ChunkConsistentGDN, borrows Megatron's weights.

        The ``kv_cache`` tensors vLLM allocates for this layer are deliberately UNUSED. Under
        chunk-consistent decode the recurrent state lives on the chunk grid, and every read and write
        of it goes through ChunkConsistentGDN. What we actually need from vLLM is the bookkeeping the
        state allocation buys: a per-request slot id and the prefill/decode split, delivered as
        ``GDNAttentionMetadata``.
        """

        def __init__(self, *, vllm_config, prefix: str, gdn):
            super().__init__()
            self.prefix = prefix
            # A plain list, not an attribute: assigning `gdn` directly would register Megatron's
            # GatedDeltaNet as a submodule of this layer and duplicate every parameter.
            self._gdn = [gdn]

            self.model_config = vllm_config.model_config
            self.cache_config = vllm_config.cache_config
            self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            if vllm_config.speculative_config:
                raise ValueError(
                    "[zerokl-gdn] speculative decoding is incompatible with chunk-consistent decode"
                )

            # Megatron shards GDN heads over ITS tensor-parallel group; vLLM must agree, which is why
            # the zero-KL recipe pins Megatron TP == inference TP.
            self.tp_size = gdn.tp_size
            self.num_k_heads = gdn.num_key_heads
            self.num_v_heads = gdn.num_value_heads
            self.head_k_dim = gdn.key_head_dim
            self.head_v_dim = gdn.value_head_dim
            self.conv_kernel_size = gdn.conv_kernel_dim

            self.kv_cache = (torch.tensor([]), torch.tensor([]))
            self._cc = None

            ctx = vllm_config.compilation_config.static_forward_context
            if prefix in ctx:
                raise ValueError(f"Duplicate layer name: {prefix}")
            ctx[prefix] = self

        # -- MambaBase surface -----------------------------------------------------------
        @property
        def mamba_type(self):
            return MambaAttentionBackendEnum.GDN_ATTN

        def get_state_shape(self):
            return MambaStateShapeCalculator.gated_delta_net_state_shape(
                self.tp_size, self.num_k_heads, self.num_v_heads,
                self.head_k_dim, self.head_v_dim, self.conv_kernel_size, 0,
            )

        def get_state_dtype(self):
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )

        # -- the core --------------------------------------------------------------------
        def _state(self):
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            from .gdn_chunk_consistent import ChunkConsistentGDN

            gdn = self._gdn[0]
            # Re-read every call: native weight sync rebinds `.data`, and a captured tensor would pin
            # the pre-update weights and silently roll back the policy.
            conv_weight = gdn.conv1d.weight.squeeze(1)          # [D, W]
            conv_bias = gdn.conv1d.bias if gdn.conv_bias else None

            if self._cc is None:
                self._cc = ChunkConsistentGDN(
                    capacity=self.max_num_seqs,
                    chunk_size=FLA_CHUNK_SIZE,
                    conv_weight=conv_weight,
                    conv_bias=conv_bias,
                    A_log=gdn.A_log,
                    dt_bias=gdn.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    num_v_heads=self.num_v_heads // self.tp_size,
                    head_v_dim=self.head_v_dim,
                    activation=gdn.activation,
                    dtype=conv_weight.dtype,
                    device=conv_weight.device,
                )
                logger.info("[zerokl-gdn] %s: chunk-consistent decode, capacity=%d, chunk=%d",
                            self.prefix, self.max_num_seqs, FLA_CHUNK_SIZE)
            self._cc.conv_weight, self._cc.conv_bias = conv_weight, conv_bias
            self._cc.A_log, self._cc.dt_bias = gdn.A_log, gdn.dt_bias
            return self._cc

        @torch.no_grad()
        def forward(self, mixed_qkv, a, b):
            """``mixed_qkv [T, D]`` pre-conv, ``a``/``b`` ``[T, Hv]`` -> ``o [T, Hv, Dv]`` or None."""
            from .gdn_engine_patch import chunk_consistent_core, gdn_metadata

            md = gdn_metadata(self.prefix)
            if md is None:  # V1 profiling run: no state, and nothing to warm (configs are pinned)
                return None
            return chunk_consistent_core(self._state(), md, mixed_qkv, a, b)

    _state_cls = ZeroKLGDNStateLayer
    return _state_cls


def _gdn_inference_forward(self, hidden_states, attention_mask=None, **kwargs):
    """Replacement ``GatedDeltaNet.forward`` for the in-vLLM GPTModel. Returns ``(out, bias)``.

    Mirrors Megatron's own forward step for step (``in_proj`` -> split -> conv/chunk core ->
    ``_apply_gated_norm`` -> ``out_proj``), with the conv+chunk core served by ChunkConsistentGDN
    instead of one full-sequence chunk call. CP/SP are not supported here (the zero-KL recipe runs
    CP=1), so the all-to-alls Megatron does around the core are skipped rather than faked.
    """
    if self.cp_size != 1 or self.sp_size != 1:
        raise NotImplementedError("[zerokl-gdn] in-vLLM GDN requires cp_size == sp_size == 1")

    # vLLM hands the wrapper [total_tokens] and it reshapes to Megatron sbhd [T, b=1, H].
    s, b, _ = hidden_states.shape
    if b != 1:
        raise NotImplementedError(f"[zerokl-gdn] in-vLLM GDN expects batch 1, got {b}")

    qkvzba, _ = self.in_proj(hidden_states)          # [T, 1, in_proj_dim]
    qkvzba = qkvzba.transpose(0, 1)                  # [1, T, ...]
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [self.qk_dim_local_tp * 2 + self.v_dim_local_tp, self.v_dim_local_tp,
         self.num_value_heads // self.tp_size, self.num_value_heads // self.tp_size],
        dim=-1,
    )

    core = self._zerokl_state.forward(qkv[0], alpha[0].contiguous(), beta[0].contiguous())
    if core is None:  # profiling run: shape-correct zeros, no state touched
        core = qkv.new_zeros(s, self.num_value_heads // self.tp_size, self.value_head_dim)

    core = core.unsqueeze(0)                                          # [1, T, Hv, Dv]
    gate = gate.reshape(1, s, -1, self.value_head_dim)
    norm_out = self._apply_gated_norm(core, gate)
    norm_out = norm_out.reshape(1, s, -1).transpose(0, 1).contiguous()  # back to sbhd
    return self.out_proj(norm_out)


def swap_gdn_core(gpt_modules, *, vllm_config) -> int:
    """Attach a vLLM state layer to every Megatron ``GatedDeltaNet`` and swap in the inference path.

    Returns the number of GDN layers swapped. Must run BEFORE vLLM allocates the KV cache, i.e.
    during model construction, so the state layers are in ``static_forward_context`` when the KV
    cache manager enumerates them.
    """
    if os.environ.get("SKYRL_ZEROKL_GDN") != "1":
        return 0

    from .gdn_batch_invariant import pin_fla_autotune_configs, pin_gdn_rmsnorm_rows_per_block

    pin_fla_autotune_configs()
    pin_gdn_rmsnorm_rows_per_block()

    cls = state_layer_cls()
    n = 0
    for layer in getattr(gpt_modules.decoder, "layers", []):
        gdn = getattr(layer, "self_attention", None)
        if gdn is None or not _is_gdn(gdn):
            continue
        layer_id = getattr(layer, "layer_number", n + 1) - 1
        prefix = f"decoder.layers.{layer_id}.self_attention"
        # `_zerokl_state` on a plain attribute, not a submodule: native weight sync walks
        # `gpt.named_parameters()` and must not see anything new.
        object.__setattr__(gdn, "_zerokl_state", cls(vllm_config=vllm_config, prefix=prefix, gdn=gdn))
        gdn.forward = _gdn_inference_forward.__get__(gdn, type(gdn))
        n += 1

    if n:
        print(f"[ZEROKL-GDN] swapped {n} Megatron GatedDeltaNet layer(s) -> chunk-consistent decode "
              "(vLLM-registered mamba state)", flush=True)
    return n
