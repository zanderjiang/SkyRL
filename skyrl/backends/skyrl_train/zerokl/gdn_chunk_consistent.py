"""Chunk-consistent GatedDeltaNet decode: make decode reproduce the training forward BITWISE.

THE PROBLEM. A GDN layer is trained (and prefilled) with the chunked-parallel kernel
``chunk_gated_delta_rule``, but generated with a fused *recurrent* kernel that advances the state one
token at a time. The two are algebraically equal and numerically different -- measured on
Qwen3.5-0.8B, |decode - prefill| logprobs are mean 1.7e-2 / max 0.25 with only 2.5% of tokens exact,
flat across position (a steady per-token kernel mismatch, not accumulating state drift). vLLM will
not even let you ask for batch invariance here: ``batch_invariant mode is not supported for
GDN_ATTN``. Since 3 of every 4 Qwen3.5 layers are GDN, this is the dominant term in the zero-KL
residual, not a rounding tail.

THE FIX. Don't write a decode kernel that *approximates* the chunk kernel -- decode WITH the chunk
kernel. Two measured properties make this exact (see ``gdn_batch_invariant`` and the tests):

  * PREFIX INVARIANCE: ``chunk(x[:t+1], S)[t] == chunk(x[:L], S)[t]`` bitwise. A token's output does
    not depend on tokens after it, even though the kernel tiles the chunk. (The intra-chunk ops are
    all causally row-local: ``solve_tril`` is a forward substitution, the QK^T products are
    tril-masked, and the log-decay cumsum is an inclusive scan.)
  * EXACT STATE CHAINING: running chunk-by-chunk and carrying ``final_state`` reproduces one long
    call bitwise. (The one op that reads a whole chunk -- the inter-chunk state advance, which
    rescales by ``exp(g_last)`` -- only ever runs on a FULL chunk.)

So: pin the recurrent state to the **chunk grid** (absolute positions that are multiples of C, the
same grid the trainer's single full-sequence call uses); keep the <=C tokens of the currently OPEN
chunk; and at every decode step re-run the chunk kernel over the open chunk starting from the
boundary state. Take the last row. When the open chunk fills, its ``final_state`` becomes the new
boundary state and the buffer resets.

WHAT IS BUFFERED, AND WHY PRE-CONV. We keep the open chunk's *pre-conv* ``mixed_qkv`` (plus the
gating inputs a, b) and the conv state at the boundary, then re-run the conv over the open chunk each
step with :func:`gdn_ops.gdn_causal_conv`. Buffering post-conv values would be cheaper, but vLLM's
prefill conv (``causal_conv1d_fn``) and decode conv (``causal_conv1d_update``) are different kernels
that do not agree bitwise. Re-running one invariant conv over the open chunk means decode literally
re-executes the prefill code path, so parity is by construction rather than by luck.

``q`` for past rows is never needed (``chunk_o`` reads only row t of q), but we recompute it anyway:
it falls out of the same conv, and correctness beats micro-optimisation here.

COST. The open chunk holds 1..C tokens (mean (C+1)/2), so a decoded token costs ~C/2 token-rows of
GDN work instead of 1 -- on the GDN layers only, which are cheap linear attention. C is
``FLA_CHUNK_SIZE`` and trades decode cost against training-kernel efficiency.

MEMORY. Each concurrently-running request needs ``C x qkv_dim`` bf16 of open-chunk buffer per GDN
layer (786 KiB at C=64, qkv_dim=6144). Buffers are therefore sized by the scheduler's
``max_num_seqs``, NOT by the engine's ssm-state slot count -- vLLM allocates thousands of state slots
out of leftover KV memory, and one buffer per slot would be tens of GiB. A small LRU maps live slot
ids onto buffers; the mapping is only ever established by ``prefill``, which is also the only way a
request (re)enters a slot, so an eviction can only ever hit a slot whose request is gone. ``decode``
raises rather than guess if a slot has no buffer.

This module is engine-agnostic on purpose: :class:`ChunkConsistentGDN` owns the buffers and the
math, and is tested offline (``examples/zerokl/nightly/gdn_layer_decode_parity_test.py``) without
booting vLLM. The vLLM wiring lives separately.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from .gdn_ops import gdn_causal_conv, gdn_chunk, gdn_gate_and_beta, gdn_l2norm


class ChunkConsistentGDN:
    """Per-layer open-chunk state for one GDN layer, shared by prefill and decode.

    ``capacity`` buffers back at most that many concurrently-running requests; engine slot ids are
    mapped onto them on ``prefill``.
    """

    def __init__(
        self,
        *,
        capacity: int,
        chunk_size: int,
        conv_weight: torch.Tensor,   # [D_qkv, W]
        conv_bias: torch.Tensor | None,
        A_log: torch.Tensor,         # [Hv]
        dt_bias: torch.Tensor,       # [Hv]
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        activation: str | None = "silu",
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ):
        if num_v_heads % num_k_heads:
            raise ValueError(f"num_v_heads {num_v_heads} must be a multiple of num_k_heads {num_k_heads}")
        self.C = chunk_size
        self.conv_weight = conv_weight
        self.conv_bias = conv_bias
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.num_v_heads = num_v_heads
        self.head_v_dim = head_v_dim
        self.activation = activation

        W = conv_weight.shape[-1]
        self.qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        self.capacity = capacity
        dev = device

        # open-chunk ring: pre-conv inputs of the tokens since the last chunk boundary
        self.x_buf = torch.zeros(capacity, chunk_size, self.qkv_dim, dtype=dtype, device=dev)
        self.a_buf = torch.zeros(capacity, chunk_size, num_v_heads, dtype=dtype, device=dev)
        self.b_buf = torch.zeros(capacity, chunk_size, num_v_heads, dtype=dtype, device=dev)
        self.fill = torch.zeros(capacity, dtype=torch.int32, device=dev)
        # state AT the last chunk boundary (not at the last token)
        self.conv_state0 = torch.zeros(capacity, self.qkv_dim, W - 1, dtype=dtype, device=dev)
        self.ssm_state0 = torch.zeros(capacity, num_v_heads, head_v_dim, head_k_dim,
                                      dtype=torch.float32, device=dev)

        # engine slot id -> buffer row, most-recently-used last
        self._slot2buf: OrderedDict[int, int] = OrderedDict()
        self._free: list[int] = list(range(capacity))

    # -- slot <-> buffer mapping ---------------------------------------------------------
    def _assign(self, slot: int) -> int:
        """Buffer row for a slot that is starting a fresh prefill. Evicts an LRU slot if needed."""
        buf = self._slot2buf.pop(slot, None)
        if buf is None:
            if not self._free:
                # Every live request touched its slot this step, so the LRU entry belongs to a
                # request that has finished (or was preempted -- vLLM re-prefills those).
                _dead, buf = self._slot2buf.popitem(last=False)
            else:
                buf = self._free.pop()
        self._slot2buf[slot] = buf
        return buf

    def _lookup(self, slot: int) -> int:
        """Buffer row for a decoding slot. Must already exist -- decode never starts a sequence."""
        buf = self._slot2buf.pop(slot, None)
        if buf is None:
            raise RuntimeError(
                f"[zerokl-gdn] slot {slot} is decoding but has no open-chunk buffer. Either its "
                f"prefill never went through ChunkConsistentGDN, or more than capacity={self.capacity} "
                "requests are live at once (capacity must be >= the scheduler's max_num_seqs)."
            )
        self._slot2buf[slot] = buf  # touch: most recently used
        return buf

    # -- helpers ------------------------------------------------------------------------
    def _split_qkv(self, y: torch.Tensor):
        """post-conv [T, D] -> q,k [T, Hv, Dk] (GQA-expanded, L2-normed), v [T, Hv, Dv]."""
        T = y.shape[0]
        kd = self.num_k_heads * self.head_k_dim
        # `.contiguous()`: slices of y are non-contiguous, and l2norm_fwd does a bare .view()
        q, k, v = y[:, :kd].contiguous(), y[:, kd : 2 * kd].contiguous(), y[:, 2 * kd :].contiguous()
        q = gdn_l2norm(q.view(T, self.num_k_heads, self.head_k_dim))
        k = gdn_l2norm(k.view(T, self.num_k_heads, self.head_k_dim))
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:  # GQA: same expansion megatron does before the kernel
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        v = v.view(T, self.num_v_heads, self.head_v_dim)
        return q, k, v

    def _prep(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, conv_state: torch.Tensor | None):
        """pre-conv tokens -> (q, k, v, g, beta) for one sequence, plus the conv state after them."""
        y, new_conv_state = gdn_causal_conv(
            x, self.conv_weight, self.conv_bias,
            initial_state=conv_state, activation=self.activation, return_final_state=True,
        )
        q, k, v = self._split_qkv(y)
        g, beta = gdn_gate_and_beta(a, b, self.A_log, self.dt_bias)
        return q, k, v, g, beta, new_conv_state

    # -- public API ---------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, slot: int, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Prefill one request from position 0. Returns o [P, Hv, Dv] (bitwise == a full chunk call).

        Splits at the last chunk boundary B = floor(P/C)*C: the head [0,B) advances the boundary
        state, the tail [B,P) becomes the open chunk. Prefix invariance guarantees the concatenated
        outputs equal a single call over [0,P).
        """
        P = x.shape[0]
        C = self.C
        B = (P // C) * C
        s = self._assign(slot)

        q, k, v, g, beta, conv_after_all = self._prep(x, a, b, None)

        outs = []
        if B > 0:
            o_head, s_b = gdn_chunk(q[None, :B], k[None, :B], v[None, :B], g[None, :B], beta[None, :B],
                                    initial_state=None, output_final_state=True)
            outs.append(o_head[0])
            self.ssm_state0[s] = s_b[0].float()
            # conv state at the boundary = last W-1 pre-conv inputs before B
            _, conv_at_B = gdn_causal_conv(x[:B], self.conv_weight, self.conv_bias,
                                           activation=self.activation, return_final_state=True)
            self.conv_state0[s] = conv_at_B
        else:
            self.ssm_state0[s].zero_()
            self.conv_state0[s].zero_()

        if P > B:
            o_tail, _ = gdn_chunk(q[None, B:], k[None, B:], v[None, B:], g[None, B:], beta[None, B:],
                                  initial_state=self.ssm_state0[s : s + 1], output_final_state=False)
            outs.append(o_tail[0])

        fill = P - B
        self.fill[s] = fill
        if fill:
            self.x_buf[s, :fill] = x[B:]
            self.a_buf[s, :fill] = a[B:]
            self.b_buf[s, :fill] = b[B:]
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

    @torch.no_grad()
    def decode(self, slots: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """One decode step for N requests. x/a/b are the NEW token only. Returns o [N, Hv, Dv].

        All N open chunks are recomputed in ONE varlen chunk-kernel launch; the kernel is
        cross-sequence invariant (pinned configs), so each request's row is bitwise what it would be
        alone.
        """
        N = slots.numel()
        C = self.C
        bufs = torch.tensor([self._lookup(int(s)) for s in slots.tolist()],
                            dtype=torch.long, device=self.fill.device)

        # append the new token to each open chunk
        fills = self.fill[bufs].to(torch.long)
        if (fills >= C).any():
            raise RuntimeError("open chunk should have been rolled before decode()")
        self.x_buf[bufs, fills] = x
        self.a_buf[bufs, fills] = a
        self.b_buf[bufs, fills] = b
        fills = fills + 1
        self.fill[bufs] = fills.to(torch.int32)

        qs, ks, vs, gs, bs, lens = [], [], [], [], [], []
        for i in range(N):
            s = int(bufs[i])
            n = int(fills[i])
            q, k, v, g, beta, _ = self._prep(
                self.x_buf[s, :n], self.a_buf[s, :n], self.b_buf[s, :n], self.conv_state0[s]
            )
            qs.append(q); ks.append(k); vs.append(v); gs.append(g); bs.append(beta); lens.append(n)

        cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=x.device)
        cat = lambda ts: torch.cat(ts, dim=0).unsqueeze(0)  # noqa: E731  -> [1, sum(lens), ...]
        need_state = any(n == C for n in lens)
        o, final_state = gdn_chunk(
            cat(qs), cat(ks), cat(vs), cat(gs), cat(bs),
            initial_state=self.ssm_state0[bufs], output_final_state=need_state, cu_seqlens=cu,
        )

        out = torch.stack([o[0, int(cu[i + 1]) - 1] for i in range(N)], dim=0)

        # roll every chunk that just filled: its final_state is a point on the chunk grid
        for i in range(N):
            if lens[i] != C:
                continue
            s = int(bufs[i])
            self.ssm_state0[s] = final_state[i].float()
            _, conv_at_B = gdn_causal_conv(
                self.x_buf[s, :C], self.conv_weight, self.conv_bias,
                initial_state=self.conv_state0[s], activation=self.activation, return_final_state=True,
            )
            self.conv_state0[s] = conv_at_B
            self.fill[s] = 0
        return out
