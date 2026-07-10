"""Batched expert execution for SequentialMLP: one bmm pair instead of a 256-iteration python loop.

WHY. The zero-KL MoE recipe pins ``SequentialMLP`` (grouped GEMM is batch-variant), whose forward
loops ``num_local_experts`` times calling each expert MLP. At Qwen3.5-35B-A3B under the matched-TP
recipe (TP=8 -> DP=1 -> EP=1) that is 256 local experts x 40 layers = 10,240 python-dispatched
expert calls per forward (~20k tiny kernel launches) -- measured as the dominant cost of rollout
decode, trainer scoring AND the training pass (GPUs at ~20% util, ~150 W).

WHAT. Replace the loop with a PADDED BATCHED GEMM: scatter each expert's token block into
``[E, M_pad, h]``, run ONE ``torch.bmm`` against the stacked fc1 weights, apply the identical
glu/probs epilogue, one ``torch.bmm`` against stacked fc2 weights, gather the valid rows back in
expert order. Comm needs nothing: expert linears run with ``explicit_expert_comm=True`` (the token
dispatcher owns the combine), so the sequential loop was pure local GEMMs -- and so is this.

ZERO-KL ARGUMENT. This is NOT bitwise-equal to the sequential loop (bmm and mm are different
Triton kernels), and does not need to be: both the trainer and the engine run THIS SAME function,
so rollout == scoring == training numerics move together. What zero-KL needs is
  (1) determinism,
  (2) per-token invariance to the routing of OTHER tokens (an expert row's result must not depend
      on how many tokens landed on it or on other experts), and
  (3) invariance to the padding amount M_pad.
All three reduce to properties of vLLM's ``bmm_batch_invariant`` (registered on ``aten::bmm`` by
``enable_batch_invariant_mode`` on every CUDA platform): each (batch, row-tile) program reduces its
row independently with a fixed schedule. They are asserted empirically by
``verify_batched_experts_invariance`` below and exercised by the live gate.

GRADIENTS. The stacked weights are built IN-GRAPH each forward (``torch.stack``), so autograd
routes expert weight grads back to each ``linear_fc1/linear_fc2.weight`` exactly as the standard
path does (gradient_accumulation_fusion is off on this stack). The stack itself costs ~200 MB of
copies per layer per microbatch -- noise against the 5 s/microbatch it replaces.

Gated on ``SKYRL_ZEROKL_MOE_BATCHED`` (default ON). ``=0`` restores the sequential loop for A/B.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_orig_sequential_forward = None

# Rows per tile. A multiple of the batch-invariant bmm kernel's BLOCK_SIZE_M (128) so tiles align
# with its row-blocks. Staged rows are bounded by T + E*CAP regardless of routing skew.
_CAP = int(os.environ.get("SKYRL_ZEROKL_MOE_TILE_ROWS", "128"))


def _supported(self) -> bool:
    cfg = self.config
    return (
        self.num_local_experts > 1
        and not (cfg.fp8 or getattr(cfg, "fp4", False))
        and not getattr(cfg, "moe_apply_probs_on_input", False)
        and not getattr(cfg, "use_te_activation_func", False)
        and not cfg.bias_activation_fusion
        and cfg.gated_linear_unit
        and not cfg.add_bias_linear
    )


def _batched_experts_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
    """Drop-in ``SequentialMLP.forward``: identical math per expert, one bmm pair for all experts."""
    if not _supported(self):
        return _orig_sequential_forward(
            self, permuted_local_hidden_states, tokens_per_expert, permuted_probs
        )

    cfg = self.config
    x = permuted_local_hidden_states          # [T, h]
    probs = permuted_probs                    # [T]
    dev = x.device
    E = self.num_local_experts
    T = x.shape[0]
    if T == 0:
        return x.new_zeros(0, x.shape[-1]), None

    counts = tokens_per_expert.to(device=dev, dtype=torch.long)   # [E]

    # FIXED-CAPACITY TILES, not pad-to-max. Padding every expert to `counts.max()` makes the
    # staging buffer [E, max_count, h] -- memory that scales with ROUTING SKEW, not with tokens
    # (a skewed profiling batch asked for 32 GiB and OOMed engine init). Instead cut each expert's
    # rows into tiles of exactly CAP rows: total staged rows <= T + E*CAP, independent of skew.
    # CAP is a CONSTANT (a multiple of the bmm kernel's BLOCK_SIZE_M=128), so a token's row is
    # always computed by a bmm of the same shape -- which is what keeps the result invariant to
    # how many tokens share its expert.
    cap = _CAP
    n_tiles_e = (counts + cap - 1) // cap                          # [E]
    tile_cu = torch.zeros(E + 1, device=dev, dtype=torch.long)
    tile_cu[1:] = n_tiles_e.cumsum(0)
    n_tiles = int(tile_cu[-1])

    tok_expert = torch.repeat_interleave(torch.arange(E, device=dev), counts)          # [T]
    cu = torch.zeros(E + 1, device=dev, dtype=torch.long)
    cu[1:] = counts.cumsum(0)
    off = torch.arange(T, device=dev) - cu[:-1].repeat_interleave(counts)              # within-expert
    tile_idx = tile_cu[:-1].repeat_interleave(counts) + off // cap                     # [T]
    row_idx = off % cap                                                                # [T]
    tile_expert = torch.repeat_interleave(torch.arange(E, device=dev), n_tiles_e)      # [n_tiles]

    xp = x.new_zeros(n_tiles, cap, x.shape[-1])
    xp[tile_idx, row_idx] = x
    pp = probs.new_zeros(n_tiles, cap)
    pp[tile_idx, row_idx] = probs

    # stacked weights in bmm layout ([E, h, 2f] / [E, f, h], contiguous), in-graph for autograd;
    # gathered per tile so an expert with several tiles reuses its weights.
    w1 = torch.stack([e.linear_fc1.weight.t() for e in self.local_experts])   # [E, h, 2f]
    w2 = torch.stack([e.linear_fc2.weight.t() for e in self.local_experts])   # [E, f, h]
    w1g = w1[tile_expert]
    w2g = w2[tile_expert]

    # torch.ops.aten.bmm, NOT torch.bmm: enable_batch_invariant_mode rebinds the PYTHON attr
    # torch.bmm to its raw Triton function, which never records autograd. Dispatching the
    # OPERATOR routes Autograd -> (overridden, batch-invariant) CUDA kernel: grads AND
    # determinism, and the backward's bmms take the same deterministic kernel.
    inter = torch.ops.aten.bmm(xp, w1g)                                   # [n_tiles, cap, 2f]

    # the exact glu MLP.forward applies on the plain (no-TE, unfused) branch
    x_glu, x_linear = torch.chunk(inter, 2, dim=-1)
    if (val := cfg.activation_func_clamp_value) is not None:
        x_glu = x_glu.clamp(min=None, max=val)
        x_linear = x_linear.clamp(min=-val, max=val)
    inter = cfg.activation_func(x_glu) * (x_linear + cfg.glu_linear_offset)

    # per_token_scale epilogue, replicated including the dtype round-trip
    original_dtype = inter.dtype
    inter = inter * pp.unsqueeze(-1)
    inter = inter.to(original_dtype)

    out = torch.ops.aten.bmm(inter.contiguous(), w2g)                     # [n_tiles, cap, h]
    output_local = out[tile_idx, row_idx]                                 # [T, h], expert order
    # explicit_expert_comm: the dispatcher owns any reduction; the sequential loop did none here.
    return output_local, None


def install_batched_sequential_mlp() -> bool:
    """Rebind ``SequentialMLP.forward`` to the batched implementation. Idempotent.

    Installed from ``prepare_zerokl_moe`` on BOTH the trainer and the engine so the two runtimes
    move together. ``SKYRL_ZEROKL_MOE_BATCHED=0`` keeps megatron's sequential loop.
    """
    global _orig_sequential_forward

    if os.environ.get("SKYRL_ZEROKL_MOE_BATCHED", "1") != "1":
        return False
    from megatron.core.transformer.moe.experts import SequentialMLP

    if getattr(SequentialMLP, "_zerokl_batched", False):
        return True
    _orig_sequential_forward = SequentialMLP.forward
    SequentialMLP.forward = _batched_experts_forward
    SequentialMLP._zerokl_batched = True
    print("[ZEROKL-MOE] SequentialMLP.forward -> padded batched expert GEMMs "
          "(one bmm pair per layer instead of a per-expert python loop)", flush=True)
    return True


@torch.no_grad()
def verify_batched_experts_invariance(*, E: int = 16, h: int = 256, f: int = 64,
                                      dtype=torch.bfloat16) -> None:
    """Assert the three properties the batched path rests on. Raises on violation. GPU-only.

    (1) determinism: same inputs twice -> bitwise-equal.
    (2) routing invariance: a token's output through expert e is bitwise-identical no matter how
        many tokens share e or what other experts received.
    (3) padding invariance: changing M_pad (by inflating another expert's load) does not change
        existing rows.
    """
    dev = "cuda"
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * f, h, device=dev, dtype=dtype) * 0.05
    w2 = torch.randn(E, h, f, device=dev, dtype=dtype) * 0.05
    tok = torch.randn(h, device=dev, dtype=dtype)

    def run(counts, probe_expert, probe_slot, extra_seed):
        torch.manual_seed(extra_seed)
        T = int(sum(counts))
        cu = [0]
        for c in counts:
            cu.append(cu[-1] + c)
        x = torch.randn(T, h, device=dev, dtype=dtype)
        p = torch.rand(T, device=dev, dtype=torch.float32)
        pos = cu[probe_expert] + probe_slot
        x[pos] = tok
        p[pos] = 0.5
        M = max(counts)
        xp = x.new_zeros(E, M, h)
        pp = p.new_zeros(E, M)
        ie = torch.repeat_interleave(torch.arange(E, device=dev),
                                     torch.tensor(counts, device=dev))
        im = torch.arange(T, device=dev) - torch.tensor(cu[:-1], device=dev).repeat_interleave(
            torch.tensor(counts, device=dev))
        xp[ie, im] = x
        pp[ie, im] = p
        inter = torch.bmm(xp, w1.transpose(1, 2))
        a, b = torch.chunk(inter, 2, dim=-1)
        inter = (torch.nn.functional.silu(a) * b * pp.unsqueeze(-1)).to(dtype)
        out = torch.bmm(inter, w2.transpose(1, 2))
        return out[probe_expert, probe_slot].clone()

    base = run([4] * E, 2, 1, seed0 := 7)
    again = run([4] * E, 2, 1, seed0)
    assert torch.equal(base, again), "[zerokl] batched experts: NOT deterministic"

    skew = [1] * E
    skew[2] = 4
    skew[5] = 300          # inflate another expert -> changes M_pad AND other-batch content
    moved = run(skew, 2, 1, seed0)
    # NOTE: run() places random tokens around the probe; with different counts the surrounding
    # content differs by construction, which is exactly the point -- the probe row must not care.
    assert torch.equal(base, moved), "[zerokl] batched experts: row depends on routing/padding"
    print("[ZEROKL-MOE] batched-experts invariance verified (deterministic, routing- and "
          "padding-invariant)", flush=True)
