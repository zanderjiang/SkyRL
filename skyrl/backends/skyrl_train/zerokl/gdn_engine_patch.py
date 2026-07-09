"""Replace vLLM's GDN decode with chunk-consistent decode, so rollout == training BITWISE.

WHAT vLLM DOES TODAY. ``QwenGatedDeltaNetAttention._forward_core`` runs two different kernels for
the two phases of the same layer:

    prefill: causal_conv1d_fn      + chunk_gated_delta_rule      (chunked-parallel)
    decode:  causal_conv1d_update  + fused_*_gated_delta_rule_*  (one-token recurrent)

They are algebraically equal and numerically different. On Qwen3.5-0.8B the resulting
|decode - prefill| logprob gap is mean 1.7e-2 / max 0.25 with 2.5% of tokens exact, and vLLM refuses
``VLLM_BATCH_INVARIANT=1`` for GDN outright ("batch_invariant mode is not supported for GDN_ATTN").
Three of every four Qwen3.5 layers are GDN, so this -- not rounding -- is the zero-KL residual.

WHAT WE DO. Route BOTH phases through :class:`~.gdn_chunk_consistent.ChunkConsistentGDN`, which
decodes by re-running the *training* chunk kernel over the open chunk. See that module for why this
is exact. This file is only plumbing: pull the right slots and token ranges out of
``GDNAttentionMetadata`` and hand them over.

THE SEMANTIC CHANGE, AND WHY WE FAIL LOUD. ``ssm_state[slot]`` now means "state at the last chunk
boundary", not "state after the last token". Nothing outside this file may interpret it. Any vLLM
feature that reads, splits, or reuses that state behind our back therefore has to be off:

  * chunked prefill -- a prompt spread over several forwards would need a mid-prompt state
  * prefix caching  -- would resume a sequence from another sequence's boundary state
  * speculative decoding -- advances the state by several tokens with the spec kernels
  * CUDA graphs -- our decode is a python loop over ragged open chunks; it does not capture

We raise at engine init rather than silently degrade (``setup_envvars_for_vllm``), and assert
``spec_sequence_masks is None`` here. NOTE: prefix caching, chunked prefill and CUDA graphs were
previously *proven bitwise-safe for the softmax layers* and are worth 4.6x rollout throughput. They
are off because the GDN path cannot support them yet, not because they are unsafe. Re-enabling them
for GDN (an open-chunk-aware state cache) is a follow-up, not a bug.

Monkey-patching the class method is deliberate and is how the rest of the zero-KL stack works: the
custom op resolves ``self._forward_core`` at call time, so a class-level rebind reaches every layer.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_patched = False

# Bumped on every _forward_core call. A patch installed in the parent process does NOT reach vLLM's
# EngineCore subprocess (VLLM_ENABLE_V1_MULTIPROCESSING=1), and the failure is silent: the run looks
# patched and reproduces the baseline numbers exactly. Tests assert this is nonzero.
CALL_COUNT = 0


def forward_core_call_count() -> int:
    return CALL_COUNT

# vLLM features that would read/advance ssm_state outside chunk-consistent decode's control.
INCOMPATIBLE_ENGINE_ARGS = (
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "speculative_config",
)


def gdn_engine_patch_enabled() -> bool:
    return os.environ.get("SKYRL_ZEROKL_GDN") == "1"


def assert_engine_args_compatible(kwargs: dict) -> None:
    """Raise if any engine arg is incompatible with chunk-consistent GDN decode.

    Called from ``vllm_engine.setup_envvars_for_vllm``. Loud on purpose: each of these silently
    reinterprets ``ssm_state`` and would produce a plausible-looking, non-bitwise rollout.
    """
    bad = [k for k in INCOMPATIBLE_ENGINE_ARGS if kwargs.get(k)]
    if not kwargs.get("enforce_eager", False):
        bad.append("enforce_eager=False (CUDA graphs)")
    if bad:
        raise ValueError(
            f"[zerokl-gdn] SKYRL_ZEROKL_GDN=1 but the engine was configured with {bad}. "
            "Chunk-consistent GDN decode redefines ssm_state[slot] as the state at the last CHUNK "
            "BOUNDARY; these features read or advance it on their own terms and would break bitwise "
            "zero-KL. They are safe for the softmax layers (proven, 4.6x rollout) -- supporting them "
            "for GDN is a follow-up. Unset them or unset SKYRL_ZEROKL_GDN."
        )


def lift_gdn_batch_invariance_veto() -> None:
    """Let ``VLLM_BATCH_INVARIANT=1`` coexist with GDN. Idempotent.

    vLLM's ``get_mamba_attn_backend`` raises "batch_invariant mode is not supported for GDN_ATTN"
    because its stock decode is a recurrent kernel with no batch-invariant form. With
    chunk-consistent decode the GDN layers ARE batch-invariant (pinned autotune configs; asserted by
    ``gdn_batch_invariant.verify_gdn_batch_invariance``). Without lifting the veto the model's
    softmax-attention layers can never be made invariant either, and they carry their own ~1e-2
    decode-vs-prefill gap.

    Both engines need this: vLLM's native GDN class and the Megatron GPTModel running inside vLLM.
    ``_cached_get_mamba_attn_backend`` is ``@cache``d, so this must run before the backend is first
    resolved (KV-cache spec collection, i.e. after model init).
    """
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

    GDNAttentionBackend.supports_batch_invariance = classmethod(lambda cls: True)


def _get_layer_state(self):
    """Lazily build this layer's ChunkConsistentGDN, sized from the engine's own slot count.

    Weight tensors are re-read from the module on every call rather than captured: native weight sync
    rebinds ``.data``, and a stale reference here would silently roll back the policy update.
    """
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

    from .gdn_chunk_consistent import ChunkConsistentGDN

    conv_weight = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    conv_bias = getattr(self.conv1d, "bias", None)

    cc = getattr(self, "_zerokl_gdn", None)
    if cc is None:
        # Size by the scheduler's concurrency cap, NOT by kv_cache[1].shape[0]: vLLM turns all
        # leftover memory into ssm-state slots (thousands), and an open-chunk buffer per slot is tens
        # of GiB per layer. Only max_num_seqs of them can be live at once.
        capacity = getattr(self, "_zerokl_max_num_seqs", None) or self.kv_cache[1].shape[0]
        cc = ChunkConsistentGDN(
            capacity=capacity,
            chunk_size=FLA_CHUNK_SIZE,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            num_v_heads=self.num_v_heads // self.tp_size,
            head_v_dim=self.head_v_dim,
            activation=self.activation,
            dtype=conv_weight.dtype,
            device=conv_weight.device,
        )
        self._zerokl_gdn = cc
        mib = cc.x_buf.numel() * cc.x_buf.element_size() / 2**20
        logger.info(
            "[zerokl-gdn] %s: chunk-consistent decode, capacity=%d, chunk=%d, open-chunk buf %.0f MiB",
            self.prefix, capacity, FLA_CHUNK_SIZE, mib,
        )

    cc.conv_weight, cc.conv_bias = conv_weight, conv_bias
    cc.A_log, cc.dt_bias = self.A_log, self.dt_bias
    return cc


@torch.no_grad()
def chunk_consistent_core(cc, md, mixed_qkv, a, b):
    """Run one GDN layer's core for a vLLM batch: ``GDNAttentionMetadata`` -> ChunkConsistentGDN.

    Engine-shape plumbing only, shared by the two engines that need it: vLLM's native
    ``QwenGatedDeltaNetAttention`` (:func:`_zerokl_forward_core`) and the Megatron ``GPTModel``
    running inside vLLM (``zerokl.gdn_gptmodel``). ``mixed_qkv`` is PRE-conv (the conv is ours),
    ``a`` is the gate input and ``b`` the beta input. Returns ``o [T, Hv, Dv]``.
    """
    if md.spec_sequence_masks is not None:
        raise RuntimeError(
            "[zerokl-gdn] speculative decoding advances ssm_state with the spec kernels, which do "
            "not respect the chunk grid. Disable spec decode for bitwise zero-KL."
        )

    n_dec, n_pre = md.num_decodes, md.num_prefills
    if n_dec == 0 and n_pre == 0:
        return None
    if md.num_decode_tokens != n_dec:
        raise RuntimeError(f"[zerokl-gdn] expected 1 token per decode, got {md.num_decode_tokens} for {n_dec}")

    T = md.num_actual_tokens
    x, b, a = mixed_qkv[:T], b[:T], a[:T]
    out = torch.empty(T, cc.num_v_heads, cc.head_v_dim, dtype=x.dtype, device=x.device)

    # Non-spec token order is decode-first, then prefill (see GDNAttentionMetadata builder), and
    # output rows map 1:1 onto mixed_qkv rows.
    if n_dec:
        dec_slots = md.non_spec_state_indices_tensor[:n_dec]
        out[:n_dec] = cc.decode(dec_slots, x[:n_dec], a[:n_dec], b[:n_dec])

    if n_pre:
        # No prefix caching, no chunked prefill => every prefill starts at position 0 with no
        # inherited state. ChunkConsistentGDN.prefill() assumes exactly that.
        if md.prefill_has_initial_state is not None and bool(md.prefill_has_initial_state.any()):
            raise RuntimeError(
                "[zerokl-gdn] a prefill carries an initial state (prefix caching or chunked prefill "
                "is on). ssm_state is a chunk-boundary state under this patch and cannot be resumed."
            )
        pre_slots = md.prefill_state_indices.tolist()
        if n_dec and set(pre_slots) & set(md.non_spec_state_indices_tensor[:n_dec].tolist()):
            raise RuntimeError("[zerokl-gdn] a slot is being prefilled and decoded in one batch")
        qsl = md.prefill_query_start_loc.tolist()  # offsets INTO the prefill region
        for i, slot in enumerate(pre_slots):
            s, e = n_dec + qsl[i], n_dec + qsl[i + 1]
            out[s:e] = cc.prefill(int(slot), x[s:e], a[s:e], b[s:e])
    return out


def gdn_metadata(prefix: str):
    """This layer's ``GDNAttentionMetadata``, or None during a V1 profiling run."""
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    md = get_forward_context().attn_metadata
    if md is None:
        return None
    assert isinstance(md, dict)
    md = md[prefix]
    assert isinstance(md, GDNAttentionMetadata)
    return md


@torch.no_grad()
def _zerokl_forward_core(self, mixed_qkv, b, a, core_attn_out):
    """Drop-in ``QwenGatedDeltaNetAttention._forward_core``: one code path, both phases."""
    global CALL_COUNT

    CALL_COUNT += 1
    md = gdn_metadata(self.prefix)
    if md is None:
        # V1 profiling run. vLLM warms `self.chunk_gated_delta_rule` here so the Triton autotuner
        # doesn't OOM benchmarking after the KV cache is allocated. We never call that op (and on
        # SM90 it resolves to a JIT-compiled FlashInfer kernel, ~5 min), and with a single pinned
        # config Triton's Autotuner skips benchmarking entirely. Nothing to warm.
        return

    out = chunk_consistent_core(_get_layer_state(self), md, mixed_qkv, a, b)
    if out is not None:
        core_attn_out[: out.shape[0]] = out


def install_gdn_engine_patch(*, force: bool = False) -> bool:
    """Rebind ``QwenGatedDeltaNetAttention._forward_core``. Idempotent; no-op unless SKYRL_ZEROKL_GDN=1."""
    global _patched

    if _patched:
        return True
    if not (force or gdn_engine_patch_enabled()):
        return False

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    from .gdn_batch_invariant import (
        pin_fla_autotune_configs,
        pin_gdn_rmsnorm_rows_per_block,
    )

    pin_fla_autotune_configs()
    # The GDN layer's output norm (RMSNormGated) sizes its Triton tile from the row count, so its
    # fp32 reduction order differs between decode (M = tokens * heads = 16) and prefill (M = 3472).
    # Chunk-consistent decode is pointless while that stands.
    pin_gdn_rmsnorm_rows_per_block()

    # `get_current_vllm_config()` is a contextvar that is only set while the model is being built,
    # so capture max_num_seqs there and read it back at first forward.
    _orig_init = QwenGatedDeltaNetAttention.__init__

    def _init(self, config, vllm_config, prefix="", gqa_interleaved_layout=False):
        _orig_init(self, config, vllm_config, prefix, gqa_interleaved_layout)
        self._zerokl_max_num_seqs = vllm_config.scheduler_config.max_num_seqs

    QwenGatedDeltaNetAttention.__init__ = _init
    QwenGatedDeltaNetAttention._forward_core = _zerokl_forward_core

    lift_gdn_batch_invariance_veto()
    # Belt and braces: the packed recurrent decode fast path is what we are replacing. Our
    # _forward_core never consults this flag, but leaving it True would mislead anyone reading state.
    QwenGatedDeltaNetAttention.enable_packed_recurrent_decode = False
    _patched = True
    print("[ZEROKL-GDN] engine: QwenGatedDeltaNetAttention._forward_core -> chunk-consistent decode",
          flush=True)
    return True
