"""MoE extension of the bitwise zero-KL stack (TP=PP=EP=1, local no-TE layer spec).

The dense zero-KL recipe rests on one invariant: **every op's per-token output is independent of
how many other tokens share the batch**. vLLM's batch-invariant aten overrides give that for
mm/addmm/linear/softmax/log_softmax/rms_norm, and the ``num_splits=1`` varlen attention gives it
for attention. A Megatron MoE layer adds three ops the overrides do not reach -- routing top-k,
expert dispatch (permute) and expert combine (unpermute) -- and one of them is outright
nondeterministic. This module supplies the missing pieces:

1. ``force_zerokl_moe_config``  -- pins the provider onto the only batch-invariant MoE recipe:
   SequentialMLP (``moe_grouped_gemm=False``, plain ``F.linear`` per expert, so the aten override
   applies), the allgather dispatcher (a no-op at TP=EP=1), fp32 router, no token dropping, and
   every MoE fusion off (the fused permute/router kernels are TE kernels and TE is absent here).

2. ``zerokl_local_layer_spec`` -- ``get_gpt_layer_local_spec(num_experts=...)`` so the MoE MLP is
   built from Megatron-core local modules, with the model-specific SelfAttention preserved
   (OLMoE norms q/k over ``num_heads * head_dim``, not per-head, so it needs its own class).

3. ``enable_moe_deterministic_ops`` -- replaces the two batch-variant ops (see the audit below),
   gated by ``SKYRL_ZEROKL_MOE_DETERMINISTIC`` (default on when this module is used at all; set
   to ``0`` to A/B the unpatched baseline).

4. ``patch_olmoe_bridge_for_sequential_mlp`` -- megatron-bridge's OLMoE mapping only names the
   grouped-GEMM expert params, so with SequentialMLP no expert weight would load from HF.

Determinism audit (megatron-core, paths as installed under the zerokl nightly venv):

  * ``moe_utils.unpermute`` -- ``output_tokens.scatter_add_(0, sorted_indices..., permuted_tokens)``.
    VERDICT: BROKEN. With top-k>1 the destination indices repeat k times per token, so CUDA
    scatter_add_ lowers to atomicAdd: the k expert contributions to a token are summed in
    hardware-arbitrary order. Nondeterministic run-to-run and batch-variant (decode's k adds land
    in a different order than prefill's). This is the MoE combine, i.e. exactly the weighted sum of
    the top-k expert outputs. FIX: ``_fixed_order_combine`` gathers each token's k rows and adds
    them in ascending-expert order -- no atomics, no cross-token reduction.

  * ``moe_utils.topk_routing_with_score_function`` -- ``torch.topk(..., sorted=torch.is_grad_enabled())``.
    VERDICT: BROKEN when ``moe_router_pre_softmax=False``. There the softmax runs *over the top-k
    scores in the order topk returned them*, so an unsorted (grad-disabled) order sums the
    denominator's k exponentials differently than the sorted (grad-enabled) order -> different
    probs. The rollout engine runs under no_grad and the trainer's training forward under grad, so
    the two disagree by construction. Harmless when ``pre_softmax=True`` (OLMoE): probs are read
    off the full softmax and scattered by index, so top-k order never reaches an arithmetic
    reduction. FIX: force ``sorted=True`` inside the router for both grad modes.

  * ``moe_utils.permute`` -- ``routing_map.bool().T.reshape(-1).argsort(descending=True, stable=True)``
    then ``tokens.index_select``. VERDICT: SAFE. Stable sort of a bool key is a unique permutation;
    index_select is a pure gather. The permutation itself depends on the token count (it must), but
    the combine undoes it, so no per-token output depends on it.

  * ``token_dispatcher.MoEAllGatherTokenDispatcher`` -- ``local_probs = probs.T.masked_select(local_map.T)``.
    VERDICT: SAFE. masked_select emits in row-major order of the (E, T) mask, the same order permute
    lays out rows, so probs stay paired with their tokens. Deterministic gather.

  * ``experts.SequentialMLP.forward`` -- ``torch.split(permuted, tokens_per_expert)`` then a python
    loop of ``MLP``s, ``torch.cat``. VERDICT: SAFE *given* ``moe_grouped_gemm=False``. Each expert's
    GEMM goes through aten::linear/matmul, which the batch-invariant override makes independent of
    that expert's row count; the cat order is the expert order. Grouped GEMM would be batch-variant
    (its tile schedule depends on the per-expert token counts) -- hence the hard pin.

  * ``moe_utils.RouterGatingLinearFunction.forward`` -- ``te_general_gemm`` if TE else ``torch.mm``.
    VERDICT: SAFE on this stack (TE absent -> aten::mm -> batch-invariant override), and fp32 under
    ``moe_router_dtype="fp32"``.

  * ``router.TopKRouter._apply_expert_bias`` / ``moe_router_enable_expert_bias``. VERDICT: UNSAFE TO
    USE. ``expert_bias``/``local_tokens_per_expert`` are *buffers*, and native_weight_sync copies
    ``named_parameters()`` only -- the engine's routing bias would silently drift from the trainer's
    after the first optimizer step. ``force_zerokl_moe_config`` raises rather than pin it.

  * aux-loss / z-loss (``MoEAuxLossAutoScaler.apply``). VERDICT: SAFE. Forward-identity; they only
    graft a gradient. They are also skipped under no_grad, so engine-vs-trainer sees the same values.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# How many distinct (num_tokens, topk) shapes to structurally validate before trusting the
# fixed-order combine's index layout. Each validation costs one device->host sync.
_VALIDATE_FIRST_N_CALLS = 8

_orig_unpermute = None
_orig_topk_routing = None
_deterministic_enabled = False
_validated_calls = 0


def moe_deterministic_enabled() -> bool:
    """True unless explicitly disabled. Only consulted on the MoE local-spec zero-KL path."""
    return os.environ.get("SKYRL_ZEROKL_MOE_DETERMINISTIC", "1") == "1"


def provider_is_moe(provider) -> bool:
    return bool(getattr(provider, "num_moe_experts", None))


# --------------------------------------------------------------------------------------
# (1) provider config: the only batch-invariant MoE recipe
# --------------------------------------------------------------------------------------
def _set_if_present(provider, name, value, changes):
    if hasattr(provider, name):
        old = getattr(provider, name)
        if old != value:
            changes.append(f"{name}: {old!r} -> {value!r}")
        setattr(provider, name, value)


def force_zerokl_moe_config(provider, *, side: str) -> None:
    """Pin ``provider`` onto the batch-invariant MoE recipe. Must run on BOTH trainer and engine.

    ``side`` is only used for the log line ("TRAINER" / "ENGINE").
    """
    if not provider_is_moe(provider):
        return

    if getattr(provider, "moe_router_enable_expert_bias", False):
        raise ValueError(
            "[zerokl] moe_router_enable_expert_bias=True is incompatible with bitwise zero-KL: "
            "expert_bias is a buffer, and native_weight_sync copies named_parameters() only, so "
            "the engine's routing bias would diverge from the trainer's after the first step."
        )
    if getattr(provider, "moe_input_jitter_eps", None) is not None:
        raise ValueError("[zerokl] moe_input_jitter_eps must be None (it randomizes routing).")

    changes: list[str] = []
    # OLMoE's provider sets persist_layer_norm=True (a fused-kernel choice, not model math); the
    # local spec's torch norm asserts it off. Dense providers leave it False, so only MoE hits it.
    _set_if_present(provider, "persist_layer_norm", False, changes)
    # SequentialMLP: each expert is a plain MLP whose F.linear the batch-invariant aten override
    # makes independent of how many tokens routed to it. Grouped GEMM is batch-variant.
    _set_if_present(provider, "moe_grouped_gemm", False, changes)
    # allgather dispatcher: at TP=EP=1 its token_dispatch/token_combine are pure no-ops (see the
    # `tp_size > 1 or ep_size > 1` guards), so no collective touches the numerics. The alltoall
    # dispatcher additionally sorts chunks across ranks for no benefit here.
    _set_if_present(provider, "moe_token_dispatcher_type", "allgather", changes)
    _set_if_present(provider, "moe_router_dtype", "fp32", changes)
    # every MoE fusion is a TE kernel (absent on this stack) and/or batch-variant.
    _set_if_present(provider, "moe_permute_fusion", False, changes)
    _set_if_present(provider, "moe_permute_fusion_into_hybridep", False, changes)
    _set_if_present(provider, "moe_router_fusion", False, changes)
    _set_if_present(provider, "moe_enable_deepep", False, changes)
    # token dropping makes an expert's output depend on the *capacity*, i.e. on the batch size.
    _set_if_present(provider, "moe_expert_capacity_factor", None, changes)
    _set_if_present(provider, "moe_pad_expert_input_to_capacity", False, changes)
    # routing replay + forced load balancing rewrite the router's decisions.
    _set_if_present(provider, "moe_enable_routing_replay", False, changes)
    _set_if_present(provider, "moe_router_force_load_balancing", False, changes)
    _set_if_present(provider, "moe_router_force_biased", None, changes)
    _set_if_present(provider, "moe_shared_expert_overlap", False, changes)
    # EP/ETP must be 1: expert sharding changes the combine's reduction into a collective.
    _set_if_present(provider, "expert_model_parallel_size", 1, changes)
    _set_if_present(provider, "expert_tensor_parallel_size", 1, changes)

    print(
        f"[ZEROKL-{side}] MoE zero-KL recipe pinned "
        f"(experts={provider.num_moe_experts} topk={getattr(provider, 'moe_router_topk', '?')} "
        f"pre_softmax={getattr(provider, 'moe_router_pre_softmax', '?')}): "
        + ("; ".join(changes) if changes else "already conformant"),
        flush=True,
    )


# --------------------------------------------------------------------------------------
# (2) layer spec: local (no-TE) spec that keeps the model's own SelfAttention
# --------------------------------------------------------------------------------------
def make_zerokl_local_layer_spec(provider):
    """Return a ``config -> ModuleSpec`` callable for the zero-KL local spec.

    Dense providers delegate to megatron-bridge's ``local_layer_spec`` unchanged (the dense zero-KL
    path must not shift). MoE providers get ``get_gpt_layer_local_spec(num_experts=...,
    moe_grouped_gemm=False)`` -> MoELayer(TopKRouter, SequentialMLP) built from local modules.

    Some providers replace ``self_attention.module`` in their own spec (OLMoE applies q/k RMSNorm
    over ``num_heads * head_dim`` rather than per-head). The local spec would silently build the
    generic ``SelfAttention`` with per-head norms -- a different model. So carry the original
    spec's ``self_attention.module`` across.
    """
    orig_spec = getattr(provider, "transformer_layer_spec", None)

    def _zerokl_local_layer_spec(config):
        from megatron.bridge.models.gpt_provider import local_layer_spec
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        if not provider_is_moe(config):
            return local_layer_spec(config)

        spec = get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            normalization=config.normalization,
        )
        attn_module = _original_self_attention_module(orig_spec, config)
        if attn_module is not None:
            spec.submodules.self_attention.module = attn_module
            print(f"[ZEROKL-SPEC] MoE local spec keeps custom SelfAttention {attn_module.__name__}", flush=True)
        return spec

    _zerokl_local_layer_spec.__name__ = "zerokl_local_layer_spec"
    return _zerokl_local_layer_spec


# Providers whose own layer spec swaps in a custom SelfAttention. Keyed by the spec function's
# __name__ so we never have to *call* the original spec -- these specs are built on
# `default_layer_spec`, which resolves to the TransformerEngine spec and would fail here.
_CUSTOM_SELF_ATTENTION_BY_SPEC = {
    # OLMoE norms q and k across `num_heads * head_dim`; the generic SelfAttention norms per-head.
    "olmoe_layer_spec": ("megatron.bridge.models.olmoe.olmoe_provider", "OLMoESelfAttention"),
}


def _original_self_attention_module(orig_spec, config):
    """The custom ``self_attention.module`` the model's own spec would have used, or None."""
    import importlib

    entry = _CUSTOM_SELF_ATTENTION_BY_SPEC.get(getattr(orig_spec, "__name__", ""))
    if entry is None:
        return None
    module_path, class_name = entry
    return getattr(importlib.import_module(module_path), class_name)


# --------------------------------------------------------------------------------------
# (3) the two batch-variant ops
# --------------------------------------------------------------------------------------
def _fixed_order_combine(permuted_tokens, sorted_indices, restore_shape):
    """Sum each token's k expert rows in ascending-expert order. No atomics, no cross-token reduce.

    ``permute`` lays rows out expert-major, so for a fixed token its k rows appear at ascending row
    positions in ascending expert order. A *stable* argsort of ``sorted_indices`` therefore groups
    each token's rows together, ascending expert within the group. Gathering column j across all
    tokens and adding the k columns in order j=0..k-1 gives every token the same summation order
    regardless of how many other tokens are in the batch -- which is the bitwise invariant.

    Returns ``None`` when the layout is not the plain "every token routes to exactly k experts"
    case (e.g. token dropping), so the caller can fall back.
    """
    global _validated_calls

    num_tokens = int(restore_shape[0])
    n = int(sorted_indices.numel())
    if num_tokens == 0 or n == 0 or n % num_tokens != 0:
        return None
    k = n // num_tokens

    order = torch.argsort(sorted_indices, stable=True)
    if _validated_calls < _VALIDATE_FIRST_N_CALLS:
        _validated_calls += 1
        expected = torch.arange(num_tokens, device=sorted_indices.device).repeat_interleave(k)
        if not torch.equal(sorted_indices[order].to(expected.dtype), expected):
            raise RuntimeError(
                "[zerokl] MoE combine: tokens do not each route to exactly topk experts "
                f"(num_tokens={num_tokens}, permuted_rows={n}). Token dropping / capacity padding "
                "must be disabled for bitwise zero-KL."
            )

    rows = order.view(num_tokens, k)
    out = permuted_tokens.index_select(0, rows[:, 0].contiguous())
    for j in range(1, k):
        out = out + permuted_tokens.index_select(0, rows[:, j].contiguous())
    return out


def _deterministic_unpermute(
    permuted_tokens,
    sorted_indices,
    restore_shape,
    probs=None,
    routing_map=None,
    fused=False,
    drop_and_pad=False,
    **kwargs,
):
    """Drop-in ``moe_utils.unpermute`` whose combine is deterministic and batch-invariant."""
    if fused or drop_and_pad:
        return _orig_unpermute(
            permuted_tokens, sorted_indices, restore_shape, probs=probs, routing_map=routing_map,
            fused=fused, drop_and_pad=drop_and_pad, **kwargs,
        )

    input_dtype = permuted_tokens.dtype
    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    out = _fixed_order_combine(permuted_tokens, sorted_indices, restore_shape)
    if out is None:
        return _orig_unpermute(
            permuted_tokens, sorted_indices, restore_shape, probs=None, routing_map=routing_map,
            fused=False, drop_and_pad=False, **kwargs,
        ).to(dtype=input_dtype)
    return out.to(dtype=input_dtype)


def _make_sorted_topk_routing(orig):
    """Force ``sorted=True`` in the router's ``torch.topk``, in both grad and no-grad forwards.

    megatron-core calls ``torch.topk(scores, k, dim=1, sorted=torch.is_grad_enabled())`` from a
    closure inside ``topk_routing_with_score_function``, so there is no argument to thread through.
    Swapping ``torch.topk`` for the duration of the routing call is the smallest intervention that
    does not fork upstream code (the forward is single-threaded).
    """
    def _sorted_topk_routing(*args, **kwargs):
        real_topk = torch.topk

        def forced(*a, **kw):
            kw["sorted"] = True
            return real_topk(*a, **kw)

        torch.topk = forced
        try:
            return orig(*args, **kwargs)
        finally:
            torch.topk = real_topk

    return _sorted_topk_routing


_matmul_invariance_lib = None


def _install_moe_matmul_invariance() -> None:
    """Install vLLM's Triton persistent-matmul overrides for mm/addmm/matmul/linear.

    On SM90+ ``enable_batch_invariant_mode`` only pins the cuBLAS workspace config, which disables
    split-K (run-to-run determinism) but does NOT make GEMMs batch-invariant: cuBLAS still selects
    a different kernel/tiling for M=1 than for M=512, so a row's result changes with batch size.
    The dense zero-KL GEMMs are all bf16, where cuBLAS happens to be row-invariant at our shapes
    (validated bitwise); the MoE router's fp32 gating mm is NOT (measured 4.3e-5 row drift at
    [T,2048]x[2048,64], gate-1c localization), and expert GEMMs run at per-expert token counts.
    vLLM's own SM80 branch installs exactly these overrides; we install them for MoE processes on
    every platform. Dense models never reach this module, so the validated dense path is untouched.
    """
    global _matmul_invariance_lib
    if _matmul_invariance_lib is not None:
        return
    from vllm.model_executor.layers import batch_invariant as bi
    from vllm.platforms import current_platform

    if current_platform.is_device_capability_family(80):
        return  # vLLM's enable_batch_invariant_mode already overrides matmuls on SM80.

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("aten::mm", bi.mm_batch_invariant, "CUDA")
    lib.impl("aten::addmm", bi.addmm_batch_invariant, "CUDA")
    lib.impl("aten::matmul", bi.matmul_batch_invariant, "CUDA")
    lib.impl("aten::linear", bi.linear_batch_invariant, "CUDA")
    _matmul_invariance_lib = lib
    print("[ZEROKL-MOE] Triton batch-invariant matmul overrides installed (mm/addmm/matmul/linear; "
          "cuBLAS is not M-invariant for the fp32 router GEMM on SM90)", flush=True)


def enable_moe_deterministic_ops() -> bool:
    """Patch the MoE combine and router top-k for bitwise decode==prefill. Idempotent."""
    global _orig_unpermute, _orig_topk_routing, _deterministic_enabled

    if _deterministic_enabled:
        return True
    if not moe_deterministic_enabled():
        print("[ZEROKL-MOE] SKYRL_ZEROKL_MOE_DETERMINISTIC=0 -> leaving scatter_add_ combine and "
              "grad-dependent top-k in place (baseline A/B; NOT bitwise)", flush=True)
        return False

    _install_moe_matmul_invariance()

    from megatron.core.transformer.moe import moe_utils, router, token_dispatcher

    _orig_unpermute = moe_utils.unpermute
    _orig_topk_routing = moe_utils.topk_routing_with_score_function
    sorted_topk_routing = _make_sorted_topk_routing(_orig_topk_routing)

    # `token_dispatcher` and `router` bind these at import, so patch every namespace that holds them.
    moe_utils.unpermute = _deterministic_unpermute
    token_dispatcher.unpermute = _deterministic_unpermute
    moe_utils.topk_routing_with_score_function = sorted_topk_routing
    router.topk_routing_with_score_function = sorted_topk_routing

    _deterministic_enabled = True
    print("[ZEROKL-MOE] deterministic ops installed: fixed-order expert combine (was CUDA "
          "scatter_add_ atomics) + sorted router top-k (was sorted=is_grad_enabled)", flush=True)
    return True


def revert_moe_deterministic_ops() -> None:
    """Restore megatron-core's originals (used by the unit test's A/B)."""
    global _deterministic_enabled, _validated_calls, _matmul_invariance_lib

    if not _deterministic_enabled:
        return
    from megatron.core.transformer.moe import moe_utils, router, token_dispatcher

    moe_utils.unpermute = _orig_unpermute
    token_dispatcher.unpermute = _orig_unpermute
    moe_utils.topk_routing_with_score_function = _orig_topk_routing
    router.topk_routing_with_score_function = _orig_topk_routing
    if _matmul_invariance_lib is not None:
        # De-register the aten overrides too, or the unit test's second "unpatched" baseline
        # would silently run on the Triton matmuls.
        _matmul_invariance_lib._destroy()
        _matmul_invariance_lib = None
    _deterministic_enabled = False
    _validated_calls = 0


# --------------------------------------------------------------------------------------
# (4) megatron-bridge OLMoE mapping: grouped-GEMM names -> SequentialMLP names
# --------------------------------------------------------------------------------------
def patch_olmoe_bridge_for_sequential_mlp() -> bool:
    """Point OLMoE's HF<->Megatron expert mappings at ``experts.local_experts.N.linear_fcX.weight``.

    megatron-bridge's OlMoEBridge only declares the grouped-GEMM parameter names
    (``experts.linear_fc1.weight{i}``), because its provider hard-codes ``moe_grouped_gemm=True``.
    Under the zero-KL SequentialMLP pin nothing would match, so every expert weight would silently
    stay at its random init. The DeepSeek/Ernie bridges already use the ``local_experts.*`` form;
    this swaps OLMoE onto it. Returns False when the bridge is unavailable (non-OLMoE stacks).
    """
    try:
        from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
        from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
        from megatron.bridge.models.olmoe import olmoe_bridge
    except Exception as e:
        logger.info("[zerokl] OLMoE bridge not importable (%s); skipping mapping patch", e)
        return False

    bridge_cls = olmoe_bridge.OlMoEBridge
    if getattr(bridge_cls, "_zerokl_seqmlp_patched", False):
        return True
    orig_mapping_registry = bridge_cls.mapping_registry

    def mapping_registry(self):
        registry = orig_mapping_registry(self)
        kept = [m for m in registry.mappings if ".mlp.experts." not in m.megatron_param]
        kept.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
            )
        )
        kept.append(
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            )
        )
        return MegatronMappingRegistry(*kept)

    bridge_cls.mapping_registry = mapping_registry
    bridge_cls._zerokl_seqmlp_patched = True
    print("[ZEROKL-MOE] OLMoE bridge expert mappings -> experts.local_experts.*.linear_fcX.weight "
          "(SequentialMLP names)", flush=True)
    return True


def prepare_zerokl_moe(provider, *, side: str) -> bool:
    """Everything the MoE zero-KL path needs on one side. Returns True when MoE was engaged.

    Call after the provider exists and before ``finalize()``/model build. No-op for dense providers,
    so the validated dense path is untouched.
    """
    if not provider_is_moe(provider):
        return False
    force_zerokl_moe_config(provider, side=side)
    enable_moe_deterministic_ops()
    return True
