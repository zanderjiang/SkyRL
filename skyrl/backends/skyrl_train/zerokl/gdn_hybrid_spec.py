"""A hybrid (GatedDeltaNet + softmax attention) Megatron layer spec built from LOCAL modules only.

Qwen3.5 is 3 GatedDeltaNet layers for every 1 softmax-attention layer. Megatron can build that, via
``get_transformer_block_with_experimental_attention_variant_spec``, but only on TransformerEngine:
that builder calls ``_get_backend_spec_provider``, which opens with

    assert config.transformer_impl == "transformer_engine", \\
        "Experimental GPT decoder block spec only supports transformer engine implementation for now."

and its GDN spec asks the backend for ``column_parallel_layer_norm_linear()`` (TE's fused
layernorm+linear), which ``LocalSpecProvider`` answers with ``None``. The zero-KL stack has no TE.

So ``make_zerokl_local_layer_spec`` fell through to megatron-bridge's flat ``local_layer_spec`` and
built a dense ``SelfAttention`` for EVERY layer. The resulting model loads none of the checkpoint's
GDN weights, generates gibberish, and -- being a consistent model of *something* -- is perfectly
bitwise decode==prefill. ``gptmodel_vllm`` now refuses to run when zero GDN layers are found.

This module builds the hybrid spec directly from local modules. It is ~40 lines of assembly rather
than a monkey-patch of three TE-only private helpers, and it produces exactly the layer that Gate 1
validates (``gdn_trainer_shim_test.py::build_layer``): ``in_proj = ColumnParallelLinear`` with a
SEPARATE ``input_layernorm``, since without TE there is nothing to fuse it into.

The un-fused layernorm is why the Qwen3.5 bridge mapping also has to be retargeted -- see
``patch_qwen35_bridge_for_local_spec``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def is_hybrid_gdn(config) -> bool:
    """True when this provider/config describes a GatedDeltaNet hybrid (Qwen3.5, Qwen3-Next)."""
    return getattr(config, "experimental_attention_variant", None) == "gated_delta_net"


def _gdn_layer_spec(config, base):
    """A local GDN layer: ``base``'s TransformerLayer with self_attention -> GatedDeltaNet."""
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec

    backend = LocalSpecProvider()
    rms_norm = config.normalization == "RMSNorm"
    spec = ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            # TE would fuse the input layernorm into in_proj (TELayerNormColumnParallelLinear).
            # Locally there is no fused module, so in_proj is a plain ColumnParallelLinear and the
            # TransformerLayer keeps its own `input_layernorm`.
            in_proj=backend.column_parallel_linear(),
            out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": False},
    )
    layer = ModuleSpec(module=base.module, submodules=type(base.submodules)(**vars(base.submodules)))
    layer.submodules.self_attention = spec
    return layer


def make_zerokl_hybrid_local_spec(config):
    """``TransformerBlockSubmodules`` with GDN on the linear-attention layers, SelfAttention elsewhere.

    Both layer kinds keep their own ``input_layernorm`` and ``pre_mlp_layernorm`` (no TE fusion).
    """
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_linear_attention_pattern,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

    if config.pipeline_model_parallel_size != 1:
        raise NotImplementedError("[zerokl-gdn] hybrid local spec is PP=1 only (no layer slicing)")

    backend = LocalSpecProvider()
    rms_norm = config.normalization == "RMSNorm"
    pattern = get_linear_attention_pattern(config)  # 1 = linear attention (GDN), 0 = softmax

    base = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        normalization=config.normalization,
    )
    layer_specs = [_gdn_layer_spec(config, base) if p == 1 else base for p in pattern]

    n_gdn = sum(pattern)
    print(f"[ZEROKL-SPEC] hybrid local spec: {n_gdn} GatedDeltaNet + {len(pattern) - n_gdn} "
          f"attention layers (no TransformerEngine)", flush=True)
    return TransformerBlockSubmodules(
        layer_specs=layer_specs, layer_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    )


# ---------------------------------------------------------------------------------------------
# bridge mapping
# ---------------------------------------------------------------------------------------------
_bridge_patched = False

# TE fuses the input layernorm into the following linear, so megatron-bridge's Qwen3.5 mapping names
# it as that linear's `layer_norm_weight`. Under the local spec the layernorms are separate modules.
_LOCAL_SPEC_RENAMES = {
    "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
    "self_attention.in_proj.layer_norm_weight": "input_layernorm.weight",
    "mlp.linear_fc1.layer_norm_weight": "pre_mlp_layernorm.weight",
}


_chunked_patched = False


def patch_chunked_mapping_index_device() -> bool:
    """Keep ``ChunkedMapping``'s shard indices on the CPU, where the HF weights are. Idempotent.

    ``ChunkedMapping.get_shard_idx`` builds index tensors with a bare ``torch.arange(...)``. The
    engine builds its GPTModel inside vLLM's model loader, which installs a default-device context
    of ``cuda`` -- so those indices come out on the GPU while ``hf_weights`` are still on the CPU,
    and ``hf_weights[idx]`` dies with

        RuntimeError: indices should be either on cpu or on the same device as the indexed tensor

    Only the GDN/Mamba conv1d and qkvzba mappings inherit this, which is why the bug appears exactly
    when the hybrid spec finally gives the bridge some GDN weights to load. The trainer never hits it
    (no default-device context).
    """
    global _chunked_patched

    if _chunked_patched:
        return True
    try:
        import torch

        from megatron.bridge.models.conversion.param_mapping import ChunkedMapping
    except Exception as e:  # pragma: no cover
        logger.info("[zerokl-gdn] ChunkedMapping unavailable (%s)", e)
        return False

    def _wrap(orig):
        def _cpu_idx(self, config, local_tp):
            return [i.cpu() if torch.is_tensor(i) else i for i in orig(self, config, local_tp)]

        return _cpu_idx

    # Each subclass (GDNConv1dMapping, MambaConv1dMapping, ...) defines its own get_shard_idx and
    # shadows the base, so patching only the base silently does nothing.
    classes = [ChunkedMapping, *ChunkedMapping.__subclasses__()]
    n = 0
    for cls in classes:
        if "get_shard_idx" in cls.__dict__:
            cls.get_shard_idx = _wrap(cls.__dict__["get_shard_idx"])
            n += 1

    _chunked_patched = True
    logger.info("[zerokl-gdn] pinned get_shard_idx to CPU indices on %d ChunkedMapping class(es)", n)
    return True


def checkpoint_is_vl_named(hf_config) -> bool:
    """True when the checkpoint stores the LM under ``model.language_model.`` (VL architecture).

    Every released Qwen3.5 checkpoint does; a hand-exported LM-only checkpoint would not. Read the
    architecture BEFORE ``maybe_force_qwen35_text_bridge`` rewrites it.
    """
    archs = list(getattr(hf_config, "architectures", []) or [])
    return any(a.endswith("ForConditionalGeneration") for a in archs)


def patch_qwen35_bridge_for_local_spec(*, hf_lm_prefix: str | None = None) -> bool:
    """Retarget the Qwen3.5 bridge's weight mapping at the no-TE local spec. Idempotent.

    Two independent mismatches, both silent -- the mapping simply matches nothing, the parameters
    keep their random init, and the model builds, runs, and talks nonsense:

    1. TE-fused layernorm names. The mapping writes
       ``decoder.layers.*.self_attention.in_proj.layer_norm_weight``; the local spec has
       ``decoder.layers.*.input_layernorm.weight``.

    2. HF prefix. ``Qwen35Bridge`` (the *text* bridge, which ``maybe_force_qwen35_text_bridge``
       selects so we get a GPTModel rather than the VL model) builds HF names as ``model.layers.*``,
       while the released checkpoints store ``model.language_model.layers.*``. Pass
       ``hf_lm_prefix="model.language_model."`` for those.

    Same class of fix as ``patch_olmoe_bridge_for_sequential_mlp``. Must run before the bridge's
    mapping registry is built, i.e. before ``AutoBridge.from_hf_pretrained``.
    """
    global _bridge_patched

    if _bridge_patched:
        return True
    if os.environ.get("SKYRL_ZEROKL_GDN") != "1":
        return False
    try:
        from megatron.bridge.models.qwen import qwen35_bridge as qb
    except Exception as e:  # pragma: no cover
        logger.info("[zerokl-gdn] qwen35_bridge unavailable (%s)", e)
        return False

    def _wrap(fn):
        def inner(hf_prefix="model.", megatron_prefix=""):
            if hf_lm_prefix and hf_prefix == "model.":
                hf_prefix = hf_lm_prefix
            mappings = fn(hf_prefix=hf_prefix, megatron_prefix=megatron_prefix)
            for m in mappings:
                mp = getattr(m, "megatron_param", None)
                if not mp:
                    continue
                for te_name, local_name in _LOCAL_SPEC_RENAMES.items():
                    if mp.endswith(te_name):
                        m.megatron_param = mp[: -len(te_name)] + local_name
                        break
            return mappings

        return inner

    n = 0
    for name in ("_get_dense_lm_mappings", "_get_moe_lm_mappings"):
        for cls in (getattr(qb, "Qwen35Bridge", None), getattr(qb, "Qwen35MoEBridge", None)):
            fn = getattr(cls, name, None) if cls else None
            if fn is None:
                continue
            setattr(cls, name, staticmethod(_wrap(fn.__func__ if hasattr(fn, "__func__") else fn)))
            n += 1

    patch_chunked_mapping_index_device()
    _bridge_patched = True
    print(f"[ZEROKL-SPEC] retargeted {n} Qwen3.5 bridge mapping table(s) at the no-TE local spec "
          f"(TE layer_norm_weight -> separate norms; hf_lm_prefix={hf_lm_prefix or 'model.'})",
          flush=True)
    return True
