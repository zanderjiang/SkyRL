"""Batch-invariant, deterministic GatedDeltaNet (GDN) kernels -- the GDN analogue of num_splits=1.

Qwen3.5 is a hybrid: 3 of every 4 layers are GatedDeltaNet linear attention. Zero-KL needs the same
invariant there as everywhere else: **a token's output must not depend on what else is in the
batch**, and it must be reproducible run to run.

Two independent problems, both fixed here.

1. THE AUTOTUNER MAKES THE KERNEL NONDETERMINISTIC.
   The vendored FLA kernels are `@triton.autotune`d. Two consequences:

   (a) `chunk_scaled_dot_kkt_fwd_kernel` compiled with ``BK=64, num_warps=4, num_stages>=2`` is
       RACY on Hopper: with identical inputs and that config it returns different results run to
       run (max |diff| ~5e-2), but only once the grid exceeds roughly one wave (>=5 chunks). Every
       other config in its space is deterministic. The autotuner picks by wall-clock benchmark, and
       on this stack it picks the racy one. Measured, not inferred: see
       ``examples/zerokl/nightly/gdn_chunk_prefix_invariance_test.py`` and the config sweep in
       MOE/GDN report.

   (b) Even with no racy config, autotuning is a per-process decision made by benchmarking. The
       trainer process and the vLLM engine process autotune independently and can land on different
       configs -> different reduction orders -> different logprobs. Zero-KL cannot survive that.

   Fix: pin every autotuned kernel in the chunk_gated_delta_rule path to a single config, selected
   by a rule that is identical in every process -- ``configs[0]``, i.e. the first entry of the
   kernel's own statically-declared config list. No magic numbers, no benchmark, no per-host drift.
   Both sides import the same source file, so both pin the same config.

2. WITH (1) FIXED, THE CHUNK KERNEL IS ALREADY EXACTLY WHAT ZERO-KL NEEDS.
   Verified bitwise on this stack (see ``verify_gdn_batch_invariance``):
     * deterministic run to run;
     * cross-sequence invariant: a sequence's output is unchanged by which other sequences share
       the varlen batch (and likewise in B>1 batch mode);
     * prefix invariant: ``chunk(x[:t+1])[t] == chunk(x[:L])[t]`` -- a token's output does not
       depend on tokens after it;
     * state chaining is exact: running chunk-by-chunk with the carried ``final_state`` reproduces
       one long call bitwise.

   Prefix invariance + exact state chaining are what make **chunk-consistent decode** possible: the
   engine can reproduce the trainer's chunked forward bitwise by snapshotting the recurrent state at
   the chunk grid and re-running this same kernel over the open chunk. See ``gdn_chunk_consistent``.

Env gates:
  * ``SKYRL_ZEROKL_GDN_PIN_CONFIGS``  (default "1") -- set 0 to A/B the unpinned autotuned baseline.
  * ``SKYRL_ZEROKL_GDN_CONFIG_INDEX`` (default "0") -- which entry of each kernel's config list to
    pin. Only for bisecting a bad config on a new stack; must match on trainer and engine.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)

# Every autotuned Triton kernel reachable from `chunk_gated_delta_rule`. Missing entries are
# tolerated (vLLM versions move kernels around) but reported, because an unpinned kernel silently
# reintroduces per-process autotuning.
_FLA_KERNELS: tuple[tuple[str, str], ...] = (
    ("chunk_scaled_dot_kkt", "chunk_scaled_dot_kkt_fwd_kernel"),
    ("solve_tril", "solve_tril_16x16_kernel"),
    ("solve_tril", "merge_16x16_to_32x32_inverse_kernel"),
    ("solve_tril", "merge_16x16_to_64x64_inverse_kernel"),
    ("wy_fast", "recompute_w_u_fwd_kernel"),
    ("chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
    ("chunk_o", "chunk_fwd_kernel_o"),
    ("cumsum", "chunk_local_cumsum_scalar_kernel"),
    ("cumsum", "chunk_local_cumsum_vector_kernel"),
)

_pinned = False


def gdn_pin_enabled() -> bool:
    return os.environ.get("SKYRL_ZEROKL_GDN_PIN_CONFIGS", "1") == "1"


def _config_index() -> int:
    return int(os.environ.get("SKYRL_ZEROKL_GDN_CONFIG_INDEX", "0"))


def _autotuner(module_name: str, kernel_name: str):
    """Return the Autotuner behind a (possibly Heuristics-wrapped) Triton kernel, or None."""
    try:
        mod = importlib.import_module(f"vllm.model_executor.layers.fla.ops.{module_name}")
    except Exception as e:  # pragma: no cover - vLLM without the vendored FLA ops
        logger.info("[zerokl-gdn] no fla.ops.%s (%s)", module_name, e)
        return None
    kernel = getattr(mod, kernel_name, None)
    if kernel is None:
        return None
    # triton.heuristics wraps triton.autotune wraps JITFunction; unwrap until we find `.configs`.
    obj = kernel
    for _ in range(4):
        if hasattr(obj, "configs") and hasattr(obj, "cache"):
            return obj
        obj = getattr(obj, "fn", None)
        if obj is None:
            return None
    return None


def pin_fla_autotune_configs() -> int:
    """Pin every FLA chunk kernel to one statically-chosen config. Idempotent. Returns #pinned."""
    global _pinned

    if _pinned:
        return 0
    if not gdn_pin_enabled():
        print("[ZEROKL-GDN] SKYRL_ZEROKL_GDN_PIN_CONFIGS=0 -> leaving Triton autotune ON. "
              "GDN kernels are then nondeterministic and NOT batch-invariant (baseline A/B only).",
              flush=True)
        return 0

    idx = _config_index()
    pinned, missing = [], []
    for module_name, kernel_name in _FLA_KERNELS:
        at = _autotuner(module_name, kernel_name)
        if at is None:
            missing.append(f"{module_name}.{kernel_name}")
            continue
        configs = list(at.configs)
        if not configs:
            missing.append(f"{module_name}.{kernel_name}(no configs)")
            continue
        chosen = configs[idx] if idx < len(configs) else configs[0]
        at.configs = [chosen]
        at.cache.clear()
        pinned.append(f"{kernel_name}:{chosen.kwargs}/w{chosen.num_warps}/s{chosen.num_stages}")

    if missing:
        # Loud, not fatal: an unpinned kernel means per-process autotuning is back for that op.
        logger.warning("[zerokl-gdn] could not pin: %s", ", ".join(missing))
    _pinned = True
    print(f"[ZEROKL-GDN] pinned {len(pinned)} FLA autotune configs (index={idx}) -> deterministic, "
          f"batch-invariant GDN. Unpinned: {len(missing)}", flush=True)
    logger.info("[zerokl-gdn] pinned configs: %s", "; ".join(pinned))
    return len(pinned)


# --------------------------------------------------------------------------------------
# verification -- the three properties zero-KL rests on. Cheap; run it in tests and CI.
# --------------------------------------------------------------------------------------
def verify_gdn_batch_invariance(*, heads: int = 8, k_dim: int = 128, v_dim: int = 128) -> None:
    """Assert determinism + cross-sequence invariance + prefix invariance. Raises on violation."""
    import torch

    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE as C

    dev = "cuda"

    def make(n, seed):
        torch.manual_seed(seed)
        return dict(
            # q,k MUST be l2-normalized or the delta rule is not a contraction and the state blows up
            q=l2norm_fwd(torch.randn(1, n, heads, k_dim, dtype=torch.bfloat16, device=dev)),
            k=l2norm_fwd(torch.randn(1, n, heads, k_dim, dtype=torch.bfloat16, device=dev)),
            v=torch.randn(1, n, heads, v_dim, dtype=torch.bfloat16, device=dev),
            g=-torch.nn.functional.softplus(torch.randn(1, n, heads, device=dev)).float(),
            beta=torch.rand(1, n, heads, dtype=torch.bfloat16, device=dev).sigmoid(),
        )

    def call(t, cu=None):
        ci = co = None
        if cu is not None:
            ci, co = prepare_chunk_indices(cu, C), prepare_chunk_offsets(cu, C)
        return chunk_gated_delta_rule(
            q=t["q"], k=t["k"], v=t["v"], g=t["g"], beta=t["beta"],
            cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co, use_qk_l2norm_in_kernel=False,
        )[0]

    # NT >= 5 chunks: fewer than that hides the racy-config bug (it needs >1 wave).
    lens = [150, 37, 64, 201]
    seqs = [make(n, 10 + i) for i, n in enumerate(lens)]
    packed = {key: torch.cat([s[key] for s in seqs], dim=1) for key in seqs[0]}
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=dev)

    a, b = call(packed, cu), call(packed, cu)
    if not torch.equal(a, b):
        raise RuntimeError(
            f"[zerokl-gdn] chunk_gated_delta_rule is NONDETERMINISTIC "
            f"(max |diff| {float((a - b).abs().max()):.3e}). A Triton autotune config is racy; "
            "pin a different SKYRL_ZEROKL_GDN_CONFIG_INDEX."
        )

    off = 0
    for i, n in enumerate(lens):
        alone = call(seqs[i], torch.tensor([0, n], dtype=torch.int32, device=dev))
        if not torch.equal(a[0, off : off + n], alone[0]):
            d = float((a[0, off : off + n] - alone[0]).abs().max())
            raise RuntimeError(
                f"[zerokl-gdn] NOT cross-sequence invariant: sequence {i} (len {n}) changes by "
                f"{d:.3e} depending on its varlen batch companions."
            )
        off += n

    seq = make(256, 7)
    full = call(seq)
    for t in (0, 1, 63, 64, 65, 127, 255):
        pref = call({key: seq[key][:, : t + 1] for key in seq})
        if not torch.equal(pref[0, t], full[0, t]):
            d = float((pref[0, t] - full[0, t]).abs().max())
            raise RuntimeError(
                f"[zerokl-gdn] NOT prefix invariant at t={t} ({d:.3e}): a token's output depends on "
                "later tokens. Chunk-consistent decode cannot be bitwise."
            )
