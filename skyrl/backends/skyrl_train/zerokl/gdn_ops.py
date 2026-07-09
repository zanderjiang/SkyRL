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


def gdn_causal_conv_batched(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Batched :func:`gdn_causal_conv`: ``x [N, T, D]``, ``initial_state [N, D, W-1]`` -> ``y [N, T, D]``.

    The SAME elementwise shifted-sum expression with a leading batch dim, so ``y[i]`` is
    bitwise-identical to ``gdn_causal_conv(x[i], ..., initial_state[i])``: every op is elementwise
    over (batch, token), fp32-accumulated in the same fixed order. This is what lets
    chunk-consistent decode run ONE conv over all open chunks instead of a per-slot python loop
    (the dominant decode cost at high concurrency) without a bitwise re-qualification of a fused
    kernel.
    """
    if x.ndim != 3:
        raise ValueError(f"gdn_causal_conv_batched expects x=[N, T, D], got {tuple(x.shape)}")
    N, T, D = x.shape
    W = weight.shape[-1]
    if weight.shape[0] != D:
        raise ValueError(f"weight {tuple(weight.shape)} incompatible with x dim {D}")

    if initial_state is None:
        pad = x.new_zeros(N, W - 1, D)
    else:
        if initial_state.shape != (N, D, W - 1):
            raise ValueError(
                f"initial_state must be [N, D, W-1]={(N, D, W - 1)}, got {tuple(initial_state.shape)}")
        pad = initial_state.transpose(1, 2).to(x.dtype)  # [N, W-1, D], oldest first
    xp = torch.cat([pad, x], dim=1)  # [N, T + W - 1, D]

    acc = torch.zeros(N, T, D, dtype=torch.float32, device=x.device)
    wf = weight.float()
    for i in range(W):
        acc = acc + wf[:, i].unsqueeze(0).unsqueeze(0) * xp[:, i : i + T].float()
    if bias is not None:
        acc = acc + bias.float().unsqueeze(0).unsqueeze(0)

    if activation in ("silu", "swish"):
        acc = acc * torch.sigmoid(acc)
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation!r}")
    return acc.to(x.dtype)


L2NORM_EPS = 1e-6


class _GdnL2NormAutograd(torch.autograd.Function):
    """vLLM's ``l2norm_fwd`` in the forward, autograd of the same expression in the backward.

    ``l2norm_fwd`` is a bare Triton launch writing into ``torch.empty_like(x)``: the result carries
    no autograd history, so backprop through it would silently deliver ZERO gradient to q and k --
    the layer would still train (v flows) and the loss would still fall, which is exactly why this is
    worth a class. Forward keeps the kernel (bitwise with the engine); backward differentiates
    ``x * rsqrt(sum(x^2) + eps)``, the expression the kernel evaluates.
    """

    @staticmethod
    def forward(ctx, x):
        from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

        with torch.no_grad():
            y = l2norm_fwd(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        with torch.enable_grad():
            xd = x.detach().float().requires_grad_(True)
            y = xd * torch.rsqrt(xd.pow(2).sum(-1, keepdim=True) + L2NORM_EPS)
            (dx,) = torch.autograd.grad(y, xd, dy.float())
        return dx.to(x.dtype)


def gdn_l2norm(x: torch.Tensor) -> torch.Tensor:
    """Row-local L2 normalisation, via the same kernel the engine and trainer both import."""
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

    if torch.is_grad_enabled() and x.requires_grad:
        return _GdnL2NormAutograd.apply(x)
    return l2norm_fwd(x)


def fla_chunk_size() -> int:
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

    return FLA_CHUNK_SIZE


def _gdn_chunk_fwd(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens):
    """The bitwise forward: vLLM's vendored FLA chunk kernel with pinned autotune configs."""
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
        # `prepare_chunk_indices` is `@tensor_cache`d on tensor IDENTITY, and vLLM recycles its
        # metadata buffers. Feeding it the caller's tensor can hand back a chunk map built for a
        # previous batch's cu_seqlens. A fresh clone forces a real recompute every call.
        cu_fresh = cu_seqlens.clone()
        chunk_indices = prepare_chunk_indices(cu_fresh, FLA_CHUNK_SIZE)
        chunk_offsets = prepare_chunk_offsets(cu_fresh, FLA_CHUNK_SIZE)

    return chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        use_qk_l2norm_in_kernel=False,  # done outside, identically on both sides
    )


def _torch_chunk_gdr_one(q, k, v, g, beta, initial_state, chunk_size):
    """Differentiable fp32 chunked delta rule for ONE sequence. q..beta: ``[1, T, H, D]``.

    This is the HuggingFace / megatron ``torch_chunk_gated_delta_rule`` reference, unchanged except
    that it takes ``initial_state`` in the kernel's ``[N, H, V, K]`` layout. It exists only to supply
    a vector-Jacobian product (see :class:`_GdnChunkAutograd`); it never runs in the forward.
    """
    q, k, v, beta, g = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g))
    _, num_heads, T, k_dim = k.shape
    v_dim = v.shape[-1]

    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = (torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (torch.nn.functional.pad(x, (0, pad)) for x in (beta, g))
    Tp = T + pad
    q = q * (k_dim**-0.5)

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, v, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    eye_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(eye_mask, 0)
    for i in range(1, chunk_size):  # forward substitution -> (I - A)^-1, strictly lower triangular
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn = attn.clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    u = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    if initial_state is None:
        state = q.new_zeros(1, num_heads, k_dim, v_dim)
    else:
        state = initial_state.transpose(-1, -2).float()  # [N,H,V,K] -> [N,H,K,V]

    strict_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    outs = []
    for i in range(Tp // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        a = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(strict_mask, 0)
        v_new = u[:, :, i] - k_cumdecay[:, :, i] @ state
        outs.append((q_i * g[:, :, i, :, None].exp()) @ state + a @ v_new)
        state = state * g[:, :, i, -1, None, None].exp() + (
            k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new

    o = torch.stack(outs, dim=2).reshape(1, num_heads, Tp, v_dim)[:, :, :T]
    return o.transpose(1, 2)  # [1, T, H, V]


def _torch_chunk_gdr(q, k, v, g, beta, initial_state, cu_seqlens, chunk_size):
    if cu_seqlens is None:
        return _torch_chunk_gdr_one(q, k, v, g, beta, initial_state, chunk_size)
    bounds = cu_seqlens.tolist()
    outs = []
    for n, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        s0 = None if initial_state is None else initial_state[n : n + 1]
        outs.append(
            _torch_chunk_gdr_one(
                q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e], s0, chunk_size
            )
        )
    return torch.cat(outs, dim=1)


class _GdnChunkAutograd(torch.autograd.Function):
    """Bitwise kernel forward + reference VJP backward.

    vLLM vendors FLA's chunk kernel for *inference*: ``ChunkGatedDeltaRuleFunction`` defines a
    ``forward`` and no ``backward``, so autograd raises the moment the trainer tries to backprop
    through it. We still want that exact kernel in the forward -- it is the whole reason decode and
    training agree bitwise -- so the forward stays untouched and the backward re-derives the
    gradient by differentiating :func:`_torch_chunk_gdr`, the fp32 torch reference for the same
    function, at the same inputs.

    That gradient is not bitwise equal to the kernel's analytic gradient (nobody's is; the kernel has
    none), and it does not need to be: zero-KL constrains the FORWARD logprobs, which set the
    importance ratio. The gradient only has to be the gradient of that forward, to fp accuracy.
    Megatron makes the same trade in ``deterministic_mode``, where the forward itself changes.

    COST: backward recomputes the layer in fp32 with a ``chunk_size``-long python loop. It is the
    slowest part of a GDN training step. If that becomes the bottleneck, the fix is a real fused
    backward, not a different forward.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state, cu_seqlens, chunk_size):
        with torch.no_grad():
            o, _ = _gdn_chunk_fwd(q, k, v, g, beta, initial_state, False, cu_seqlens)
        ctx.save_for_backward(q, k, v, g, beta, initial_state)
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, initial_state = ctx.saved_tensors
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
            o = _torch_chunk_gdr(*leaves, initial_state, ctx.cu_seqlens, ctx.chunk_size)
            grads = torch.autograd.grad(o, leaves, do.float())
        grads = [gr.to(t.dtype) for gr, t in zip(grads, (q, k, v, g, beta))]
        return (*grads, None, None, None)


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

    Under ``torch.no_grad`` (both rollout paths and the trainer's scoring forward) this is exactly
    the vLLM kernel. When a gradient is required it routes through :class:`_GdnChunkAutograd`, whose
    forward is that same kernel -- so the training forward stays bitwise equal to the rollout.
    """
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state)
    )
    if not needs_grad:
        return _gdn_chunk_fwd(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens)

    if output_final_state:
        raise NotImplementedError(
            "zerokl GDN: output_final_state is not differentiable (training never asks for it; "
            "only chunk-consistent decode does, under no_grad)."
        )
    o = _GdnChunkAutograd.apply(q, k, v, g, beta, initial_state, cu_seqlens, fla_chunk_size())
    return o, None


def gdn_gate_and_beta(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """g = -exp(A_log) * softplus(a + dt_bias) in fp32; beta = sigmoid(b). Elementwise -> invariant.

    Mirrors megatron ``GatedDeltaNet._compute_g_and_beta`` exactly (fp32 g, beta in the input dtype).
    ``A_log.exp()`` is taken in the parameter's own dtype, not upcast first: megatron stores A_log in
    ``params_dtype`` (bf16) and exponentiates before the fp32 multiply, and ``exp(bf16(x)).float()``
    is not ``exp(float(x))``. Zero-KL lives in that last ulp.
    """
    g = -A_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = b.sigmoid()
    return g, beta
