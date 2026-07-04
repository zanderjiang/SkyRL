"""Element-wise weight fingerprints for the zero-KL BISECT diagnostic.

The original BISECT dumps compared per-tensor ABS-SUM (sum of |w|). That is collision-prone: two
tensors that differ element-wise can share an abs-sum (e.g. a sign flip, or a +d/-d pair), so an
abs-sum "match" does NOT prove the engine's generation weights are byte-identical to the trainer's
scoring weights. This module emits THREE independent reductions per tensor:

    absn = sum(|w|)                          # original (kept for back-compat with old dumps)
    sq   = sum(w*w)                           # changes under any magnitude perturbation
    dot  = sum(w * ((i+1) mod 9973))          # position-weighted -> changes if any element moves/changes

A match on all three (engine vs trainer, same native name/shape => same element order) makes a
real per-element difference astronomically unlikely. Everything is computed on the BF16 view (the
resident inference dtype) in float64 so the trainer's fp32 master and the engine's bf16 copy are
compared on the SAME basis -- the question is "are the bytes the engine GENERATES with equal to the
bytes the trainer SCORES with", and both sides live as bf16.

No Date/random use (safe under workflow replay). The ramp is deterministic from torch.arange.
"""
from __future__ import annotations

import torch

_RAMP_MOD = 9973  # a prime; keeps the position weight bounded so fp64 stays exact-ish


def tensor_fingerprint(t: torch.Tensor) -> tuple[float, float, float]:
    """Return (abs_sum, sum_sq, ramp_dot) of ``t`` cast to bf16 then float64."""
    flat = t.detach().to(torch.bfloat16).reshape(-1).double()
    n = flat.numel()
    absn = float(flat.abs().sum())
    sq = float((flat * flat).sum())
    ramp = (torch.arange(1, n + 1, device=flat.device, dtype=torch.float64) % _RAMP_MOD)
    dot = float((flat * ramp).sum())
    return absn, sq, dot


def fingerprint_line(name: str, t: torch.Tensor) -> str:
    """One tab-separated dump line: name, abs_sum, sum_sq, ramp_dot, shape, dtype."""
    absn, sq, dot = tensor_fingerprint(t)
    return f"{name}\t{absn:.8f}\t{sq:.8f}\t{dot:.8f}\t{tuple(t.shape)}\t{t.dtype}\n"
