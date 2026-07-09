"""Shared, provably batch-invariant GDN ops -- ONE implementation used by the engine and the trainer.

Zero-KL's core principle: the two runtimes must execute the *same* code for every op, and that code
must be batch/context-invariant. For GatedDeltaNet that means three pieces:

1. ``gdn_causal_conv`` -- the short (width-4) causal depthwise conv in front of q/k/v.

   vLLM uses two DIFFERENT kernels here: ``causal_conv1d_fn`` at prefill and
   ``causal_conv1d_update`` at decode. They do not agree bitwise, and Megatron uses a third
   (FLA's ``causal_conv1d``, or ``F.conv1d`` in deterministic mode). Rather than pick one fused
   kernel and hope it is prefix-invariant, we express the conv as what it actually is: a sum of
   ``width`` shifted, scaled copies of the input, plus bias, plus the activation. Every operation
   is elementwise, so the result for token t depends on tokens t-3..t and NOTHING else -- no
   reduction order to vary, no tiling, no batch dependence. Invariance holds by construction, not
   by measurement. The conv is a rounding error of the layer's cost (a 4-tap depthwise conv against
   two big GEMMs), so the fused kernels buy nothing worth this risk.

   Accumulation is done in fp32 and rounded once, which is both more accurate than a bf16 chain and
   -- more importantly -- fixed, so both runtimes round identically.

2. ``gdn_l2norm`` -- q/k L2 normalisation. Delegates to vLLM's ``l2norm_fwd`` (row-local; used by
   both sides) so the trainer and engine share the exact kernel.

3. ``gdn_chunk`` -- the chunked delta-rule kernel, with autotune configs pinned
   (see ``gdn_batch_invariant``). Deterministic, cross-sequence invariant, prefix invariant.

Together these make a GDN layer's per-token output independent of batching, which is what lets
chunk-consistent decode (``gdn_chunk_consistent``) reproduce the training forward bitwise.
"""

from __future__ import annotations

import torch


def gdn_causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    return_final_state: bool = False,
):
    """Width-`W` causal depthwise conv over a token sequence. Batch/prefix invariant by construction.

    Args:
        x: ``[T, D]`` input tokens (a single sequence, or one chunk of one).
        weight: ``[D, W]`` depthwise taps; ``weight[:, -1]`` multiplies the current token.
        bias: ``[D]`` or None.
        initial_state: ``[D, W-1]`` the previous ``W-1`` inputs (oldest first), or None for a fresh
            sequence (equivalent to zeros).
        activation: ``"silu"`` / ``"swish"`` or None.
        return_final_state: also return the ``[D, W-1]`` state after consuming ``x``.

    Returns:
        ``y [T, D]``, and ``final_state [D, W-1]`` when requested.

    Every op below is elementwise over the token axis, so ``y[t]`` is a pure function of
    ``x[t-W+1 .. t]``; slicing a prefix, changing the batch, or splitting the sequence into chunks
    cannot change it.
    """
    if x.ndim != 2:
        raise ValueError(f"gdn_causal_conv expects x=[T, D], got {tuple(x.shape)}")
    T, D = x.shape
    W = weight.shape[-1]
    if weight.shape[0] != D:
        raise ValueError(f"weight {tuple(weight.shape)} incompatible with x dim {D}")

    if initial_state is None:
        pad = x.new_zeros(W - 1, D)
    else:
        if initial_state.shape != (D, W - 1):
            raise ValueError(f"initial_state must be [D, W-1]={(D, W - 1)}, got {tuple(initial_state.shape)}")
        pad = initial_state.transpose(0, 1).to(x.dtype)  # [W-1, D], oldest first
    xp = torch.cat([pad, x], dim=0)  # [T + W - 1, D]

    # y[t] = sum_i w[:, i] * xp[t + i]   (i = 0..W-1; i = W-1 is the current token)
    acc = torch.zeros(T, D, dtype=torch.float32, device=x.device)
    wf = weight.float()
    for i in range(W):
        acc = acc + wf[:, i].unsqueeze(0) * xp[i : i + T].float()
    if bias is not None:
        acc = acc + bias.float().unsqueeze(0)

    if activation in ("silu", "swish"):
        acc = acc * torch.sigmoid(acc)
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation!r}")

    y = acc.to(x.dtype)
    if not return_final_state:
        return y
    final_state = xp[T:].transpose(0, 1).contiguous()  # last W-1 inputs, [D, W-1]
    return y, final_state


def gdn_l2norm(x: torch.Tensor) -> torch.Tensor:
    """Row-local L2 normalisation, via the same kernel the engine and trainer both import."""
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

    return l2norm_fwd(x)


def gdn_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
):
    """`chunk_gated_delta_rule` with pinned configs. q/k must already be L2-normalised.

    Returns ``(o, final_state)``. ``final_state`` is meaningful only when the trailing chunk of each
    sequence is FULL -- for a partial chunk it is the state after that partial chunk, which is not a
    point on the chunk grid.
    """
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

    from .gdn_batch_invariant import pin_fla_autotune_configs

    pin_fla_autotune_configs()  # idempotent; must be in effect before the first launch

    chunk_indices = chunk_offsets = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE)

    return chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        use_qk_l2norm_in_kernel=False,  # done outside, identically on both sides
    )


def gdn_gate_and_beta(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """g = -exp(A_log) * softplus(a + dt_bias) in fp32; beta = sigmoid(b). Elementwise -> invariant.

    Mirrors megatron ``GatedDeltaNet._compute_g_and_beta`` exactly (fp32 g, beta in the input dtype).
    """
    g = -torch.exp(A_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = b.sigmoid()
    return g, beta
