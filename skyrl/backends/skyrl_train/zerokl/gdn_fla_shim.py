"""Make Megatron's GatedDeltaNet execute the *engine's* GDN ops, by faking the ``fla`` package.

WHY THIS EXISTS. ``megatron/core/ssm/gated_delta_net.py`` opens with

    try:
        from fla.modules.convolution import causal_conv1d
        from fla.modules.l2norm import l2norm
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        HAVE_FLA = True
    except ImportError:
        HAVE_FLA = False

and ``GatedDeltaNet.__init__`` raises when ``HAVE_FLA`` is False. ``flash-linear-attention`` is not
installed in the zerokl venv (and we do not want it to be: a second copy of the chunk kernel is a
second set of autotune decisions, which is exactly the divergence zero-KL is built to eliminate).
Megatron's ``deterministic_mode`` fallback, ``torch_chunk_gated_delta_rule``, is not an option
either -- it asserts ``cu_seqlens is None``, i.e. it cannot run the packed (thd) path SkyRL trains on.

WHAT WE DO. Register a facade ``fla`` package in ``sys.modules`` whose three symbols forward to
``zerokl.gdn_ops``. The trainer then runs *literally the same functions* as the rollout engine:

    fla.ops.gated_delta_rule.chunk_gated_delta_rule -> gdn_ops.gdn_chunk       (pinned autotune)
    fla.modules.l2norm.l2norm                       -> gdn_ops.gdn_l2norm      (vLLM l2norm_fwd)
    fla.modules.convolution.causal_conv1d           -> gdn_ops.gdn_causal_conv (elementwise)

That is the whole zero-KL principle applied to GDN: one implementation, two runtimes.

TWO EXTRA PATCHES, both for the same reason -- "same code" has to mean same *executed* code:

  * ``_compute_g_and_beta`` and ``_prepare_qkv_for_gated_delta_rule`` are decorated with megatron's
    ``jit_fuser``, which on torch >= 2.2 is ``torch.compile``. A compiled ``exp``/``softplus`` is
    Triton's ``libdevice`` version, not ATen's, and the two disagree in the last ulp. The engine
    calls the eager ops. So we rebind both methods to eager equivalents.
    (``SKYRL_ZEROKL_GDN_EAGER_PREP=0`` restores the compiled ones, for A/B.)

  * ``causal_conv1d`` must not convolve across a packed-sequence boundary. Megatron hands us the
    whole packed row plus ``cu_seqlens``; we slice and convolve each sequence independently.

INSTALL ORDER. ``megatron.core.ssm.gated_delta_net`` binds ``chunk_gated_delta_rule`` at *import*
time, so :func:`install_fla_shim` must run before anything imports megatron.bridge / megatron.core.
It is called from ``zerokl/__init__.py`` (which ``megatron_worker.py`` imports before
``from megatron.bridge import AutoBridge``) and from ``zerokl/gptmodel_vllm.py``. Both are gated on
``SKYRL_ZEROKL_GDN=1``; the function is idempotent and safe to call again.
"""

from __future__ import annotations

import importlib.machinery
import logging
import os
import sys
import types

logger = logging.getLogger(__name__)

_installed = False


def gdn_enabled() -> bool:
    return os.environ.get("SKYRL_ZEROKL_GDN") == "1"


def _eager_prep_enabled() -> bool:
    return os.environ.get("SKYRL_ZEROKL_GDN_EAGER_PREP", "1") == "1"


# ---------------------------------------------------------------------------------------------
# adapters: megatron's call signatures -> gdn_ops
# ---------------------------------------------------------------------------------------------
def _shim_chunk_gated_delta_rule(
    query,
    key,
    value,
    g=None,
    beta=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    **_ignored,
):
    """``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` -> :func:`gdn_ops.gdn_chunk`.

    Megatron calls this positionally for q/k/v and by keyword for the rest. It never asks for
    in-kernel L2 norm (it normalises in ``_prepare_qkv_for_gated_delta_rule``), and never passes a
    custom ``scale``; both are rejected rather than silently ignored, because either one would make
    the trainer and the engine compute different things.
    """
    from .gdn_ops import gdn_chunk, gdn_l2norm

    if scale is not None:
        raise NotImplementedError("zerokl GDN shim: custom `scale` would diverge from the engine")
    if use_qk_l2norm_in_kernel:
        # The engine normalises outside the kernel too (gdn_ops.gdn_chunk passes False), so route
        # through the same l2norm rather than the kernel's internal one.
        query, key = gdn_l2norm(query), gdn_l2norm(key)
    return gdn_chunk(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )


def _no_recurrent(*_a, **_k):
    """`fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` -- present only to fail loudly.

    The recurrent kernel is exactly what chunk-consistent decode replaces: it is algebraically equal
    to the chunk kernel and numerically different (mean 1.7e-2 in logprob). Nothing in the zero-KL
    path may call it.
    """
    raise NotImplementedError(
        "[zerokl-gdn] the fused recurrent delta-rule kernel is not batch-invariant with the chunk "
        "kernel used at training/prefill. Use zerokl.gdn_chunk_consistent.ChunkConsistentGDN."
    )


def _shim_l2norm(x, dim: int = -1, eps: float = 1e-6, **_ignored):
    """``fla.modules.l2norm.l2norm`` -> :func:`gdn_ops.gdn_l2norm` (row-local, last dim)."""
    from .gdn_ops import gdn_l2norm

    if dim not in (-1, x.ndim - 1):
        raise NotImplementedError(f"zerokl GDN shim: l2norm only over the last dim (got dim={dim})")
    return gdn_l2norm(x.contiguous())


def _shim_causal_conv1d(
    x,
    weight,
    bias=None,
    activation=None,
    initial_state=None,
    output_final_state: bool = False,
    cu_seqlens=None,
    **_ignored,
):
    """``fla.modules.convolution.causal_conv1d`` -> :func:`gdn_ops.gdn_causal_conv`, per sequence.

    Args mirror FLA: ``x`` is ``[B, T, D]``, ``weight`` is ``[D, W]``. Returns ``(y, final_state)``,
    with ``final_state`` ``None`` unless requested.

    With ``cu_seqlens`` (packed/thd, ``B == 1``) each sequence is convolved on its own: a width-4
    causal conv that ran across a packed boundary would leak the previous sequence's last 3 tokens
    into the next one's first 3 outputs. FLA does the same; we do it explicitly.
    """
    import torch

    from .gdn_ops import gdn_causal_conv

    if x.ndim != 3:
        raise ValueError(f"zerokl GDN shim: causal_conv1d expects x=[B, T, D], got {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"zerokl GDN shim: causal_conv1d expects weight=[D, W], got {tuple(weight.shape)}")

    def _one(seq, state):
        return gdn_causal_conv(
            seq, weight, bias, initial_state=state, activation=activation, return_final_state=True
        )

    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError("zerokl GDN shim: packed causal_conv1d requires batch == 1")
        bounds = cu_seqlens.tolist()
        if bounds[-1] != x.shape[1]:
            raise ValueError(f"cu_seqlens[-1]={bounds[-1]} != T={x.shape[1]}")
        if initial_state is not None:
            raise NotImplementedError("zerokl GDN shim: initial_state with cu_seqlens is unsupported")
        ys, states = [], []
        for s, e in zip(bounds[:-1], bounds[1:]):
            y, st = _one(x[0, s:e], None)
            ys.append(y)
            states.append(st)
        y = torch.cat(ys, dim=0).unsqueeze(0)
        final_state = torch.stack(states, dim=0) if output_final_state else None
        return y, final_state

    ys, states = [], []
    for i in range(x.shape[0]):
        y, st = _one(x[i], None if initial_state is None else initial_state[i])
        ys.append(y)
        states.append(st)
    y = torch.stack(ys, dim=0)
    final_state = torch.stack(states, dim=0) if output_final_state else None
    return y, final_state


# ---------------------------------------------------------------------------------------------
# eager replacements for megatron's torch.compile'd helpers
# ---------------------------------------------------------------------------------------------
def _eager_compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
    """Same expression as ``GatedDeltaNet._compute_g_and_beta``, minus ``@jit_fuser``.

    Kept character-for-character on purpose (``A_log.exp()`` in the parameter dtype, softplus in
    fp32): ``zerokl.gdn_ops.gdn_gate_and_beta`` implements this identical expression, so trainer and
    engine round the same way.
    """
    import torch.nn.functional as F

    g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)
    beta = beta.sigmoid()
    return g, beta


def _eager_prepare_qkv(self, qkv, gate, beta, alpha, batch, seq_len):
    """Eager copy of ``GatedDeltaNet._prepare_qkv_for_gated_delta_rule`` (no ``@jit_fuser``)."""
    import torch

    from .gdn_ops import gdn_l2norm

    query_key, value = torch.split(
        qkv,
        [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
        dim=-1,
    )
    query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
    value = value.reshape(batch, seq_len, -1, self.value_head_dim)
    if self.use_qk_l2norm:
        query_key = gdn_l2norm(query_key.contiguous())
    split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
    query, key = torch.split(query_key, [split_size, split_size], dim=2)
    if self.num_value_heads // self.num_key_heads > 1:
        repeat_factor = self.num_value_heads // self.num_key_heads
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)
    return (
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate.contiguous(),
        beta.contiguous(),
        alpha.contiguous(),
    )


def _patch_megatron_gdn_eager() -> bool:
    """Rebind the two ``jit_fuser``-decorated helpers to eager equivalents. Returns True if done."""
    try:
        from megatron.core.ssm import gated_delta_net as mg
    except Exception as e:  # pragma: no cover - megatron absent (engine-only process)
        logger.info("[zerokl-gdn] megatron.core.ssm.gated_delta_net unavailable (%s)", e)
        return False
    mg.GatedDeltaNet._compute_g_and_beta = _eager_compute_g_and_beta
    mg.GatedDeltaNet._prepare_qkv_for_gated_delta_rule = _eager_prepare_qkv
    print("[ZEROKL-GDN] megatron GatedDeltaNet: g/beta + qkv-prep run EAGER (no torch.compile)", flush=True)
    return True


# ---------------------------------------------------------------------------------------------
# the facade
# ---------------------------------------------------------------------------------------------
def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    # A module with `__spec__ = None` makes `importlib.util.find_spec(name)` raise ValueError, and
    # `transformers.utils.is_flash_linear_attention_available()` calls exactly that on `fla` while
    # importing Qwen3-Next. Give the facade a real (loader-less) spec.
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def install_fla_shim(*, force: bool = False) -> bool:
    """Register the ``fla`` facade so ``megatron.core.ssm.gated_delta_net`` imports our ops.

    No-op (returning False) unless ``SKYRL_ZEROKL_GDN=1`` or ``force``. Idempotent. Refuses to
    shadow a real ``flash-linear-attention`` install, because then the two runtimes would silently
    stop sharing a kernel.
    """
    global _installed

    if _installed:
        return True
    if not (force or gdn_enabled()):
        return False

    if "fla" in sys.modules and not getattr(sys.modules["fla"], "__zerokl_shim__", False):
        raise RuntimeError(
            "[zerokl-gdn] a real `fla` package is already imported. Uninstall flash-linear-attention: "
            "zero-KL requires the trainer and the engine to share ONE chunk-kernel implementation "
            "(and one autotune decision)."
        )

    # __version__ "0.0.0": `transformers.is_flash_linear_attention_available()` requires >= 0.2.2 and
    # falls back to the module's __version__ when there is no distribution metadata. Declaring an old
    # version makes HF answer "no FLA" and use its own torch reference, which is what we want -- its
    # Qwen3-Next modeling would otherwise import `fused_recurrent_gated_delta_rule` from this facade.
    # Megatron imports our symbols by name and never consults that check.
    fla = _module("fla", __zerokl_shim__=True, __version__="0.0.0", __path__=[])
    ops = _module("fla.ops", __path__=[])
    gdr = _module("fla.ops.gated_delta_rule", chunk_gated_delta_rule=_shim_chunk_gated_delta_rule,
                  fused_recurrent_gated_delta_rule=_no_recurrent)
    modules = _module("fla.modules", __path__=[])
    conv = _module("fla.modules.convolution", causal_conv1d=_shim_causal_conv1d)
    l2 = _module("fla.modules.l2norm", l2norm=_shim_l2norm)

    fla.ops, fla.modules = ops, modules
    ops.gated_delta_rule = gdr
    modules.convolution, modules.l2norm = conv, l2
    # `from fla.ops.gated_delta_rule import chunk_gated_delta_rule` also works via `fla.ops`
    ops.chunk_gated_delta_rule = _shim_chunk_gated_delta_rule
    modules.causal_conv1d, modules.l2norm_fn = _shim_causal_conv1d, _shim_l2norm

    _installed = True
    print("[ZEROKL-GDN] installed `fla` shim -> zerokl.gdn_ops (chunk / l2norm / causal_conv1d)", flush=True)

    if _eager_prep_enabled():
        _patch_megatron_gdn_eager()

    # Qwen3.5 normalises with rms(x) * (1 + w); the no-TE torch norm asserts against that flag and
    # would abort while building the first layer, in the trainer and in the in-vLLM GPTModel alike.
    from .zero_centered_norm import install_zero_centered_torch_norm

    install_zero_centered_torch_norm()

    # Pin the Triton autotune configs now if vLLM is importable in this process; `gdn_chunk` pins
    # again (idempotently) before its first launch, so a deferred pin is safe, never skipped.
    try:
        from .gdn_batch_invariant import pin_fla_autotune_configs

        pin_fla_autotune_configs()
    except Exception as e:  # pragma: no cover - vLLM not importable yet
        logger.info("[zerokl-gdn] deferring autotune pin to first gdn_chunk call (%s)", e)
    return True
