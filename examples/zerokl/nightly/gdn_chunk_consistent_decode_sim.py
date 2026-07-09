"""Simulate chunk-consistent GDN decode and check it is BITWISE equal to the prefill/training pass.

Design under test (no custom kernel; reuse the training chunk kernel):

    * the recurrent state is snapshotted ONLY at absolute positions that are multiples of C
      (C = FLA_CHUNK_SIZE = 64) -- the same chunk grid the trainer's single full-sequence call uses;
    * the <=C tokens of the currently OPEN chunk are kept as post-conv k, v, g, beta;
    * each decode step appends the new token and re-runs `chunk_gated_delta_rule` over the open
      chunk with initial_state = S_B (the boundary state), taking the last row as the output;
    * when the open chunk fills to C, its `final_state` becomes the new S_B and the buffer resets.

`q` for past rows is irrelevant (chunk_o's row t reads only q row t), but we keep the real q so the
simulation matches what a real implementation would pass.

Sections:
  A. single-sequence decode simulation vs one full prefill call
  B. BATCHED varlen decode: N requests with DIFFERENT open-chunk fills decoded together, vs each
     one's full prefill -- this is the batch-invariance property the engine actually needs
  C. cost: how many extra token-rows the recompute costs vs a fused recurrent step

Run:  CUDA_VISIBLE_DEVICES=<gpu> uv run --isolated --extra zerokl \
        python examples/zerokl/nightly/gdn_chunk_consistent_decode_sim.py
Exit 0 iff every comparison is bitwise (max abs diff == 0.0).
"""

import os
import sys

import torch

sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")

HEADS = int(os.environ.get("GDN_HEADS", "32"))
K_DIM = int(os.environ.get("GDN_K", "128"))
V_DIM = int(os.environ.get("GDN_V", "128"))
DTYPE = getattr(torch, os.environ.get("GDN_DTYPE", "bfloat16"))
C = 64  # FLA_CHUNK_SIZE


def chunk(q, k, v, g, beta, initial_state=None, output_final_state=False, cu_seqlens=None):
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

    ci = co = None
    if cu_seqlens is not None:
        from vllm.model_executor.layers.fla.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        ci = prepare_chunk_indices(cu_seqlens, C)
        co = prepare_chunk_offsets(cu_seqlens, C)
    return chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=ci, chunk_offsets=co,
        use_qk_l2norm_in_kernel=False,
    )


def maxdiff(a, b):
    """NaN-safe: `max(0.0, nan)` returns 0.0 in Python, which would silently hide a NaN as 'exact'."""
    d = (a.float() - b.float()).abs().max()
    if torch.isnan(d):
        return float("nan")
    return float(d)


def make_seq(L, seed):
    """Realistic GDN inputs.

    q,k MUST be L2-normalized (megatron does it outside the kernel, so we pass
    use_qk_l2norm_in_kernel=False everywhere). Without it, (I - beta*k*k^T) is not a contraction and
    the recurrent state diverges -- state |max| ~1e24 after a few chunks, then inf/NaN. That is a
    property of the inputs, not of the kernel.
    """
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

    torch.manual_seed(seed)
    d = "cuda"
    q = l2norm_fwd(torch.randn(1, L, HEADS, K_DIM, dtype=DTYPE, device=d))
    k = l2norm_fwd(torch.randn(1, L, HEADS, K_DIM, dtype=DTYPE, device=d))
    return (
        q,
        k,
        torch.randn(1, L, HEADS, V_DIM, dtype=DTYPE, device=d),
        -torch.nn.functional.softplus(torch.randn(1, L, HEADS, device=d)).float(),
        torch.rand(1, L, HEADS, dtype=DTYPE, device=d).sigmoid(),
    )


class OpenChunkState:
    """The per-request decode state: boundary state S_B + the open chunk's post-conv tokens."""

    def __init__(self, s_b, buf):
        self.s_b = s_b            # [1, H, V, K] fp32, state at the last multiple-of-C position
        self.buf = buf            # dict of k,v,g,beta,q each [1, j, ...]; j = open-chunk fill

    @classmethod
    def from_prefill(cls, q, k, v, g, beta):
        """Prefill P tokens; snapshot the state at B = floor(P/C)*C, keep [B,P) as the open chunk."""
        P = q.shape[1]
        B = (P // C) * C
        if B == 0:
            s_b = torch.zeros(1, HEADS, V_DIM, K_DIM, dtype=torch.float32, device=q.device)
        else:
            _, s_b = chunk(q[:, :B], k[:, :B], v[:, :B], g[:, :B], beta[:, :B],
                           output_final_state=True)
        buf = {"q": q[:, B:P], "k": k[:, B:P], "v": v[:, B:P], "g": g[:, B:P], "beta": beta[:, B:P]}
        return cls(s_b, buf)

    def append(self, q1, k1, v1, g1, beta1):
        for name, t in (("q", q1), ("k", k1), ("v", v1), ("g", g1), ("beta", beta1)):
            self.buf[name] = torch.cat([self.buf[name], t], dim=1)

    @property
    def fill(self):
        return self.buf["k"].shape[1]

    def roll_if_full(self, final_state):
        if self.fill == C:
            self.s_b = final_state
            for name in self.buf:
                self.buf[name] = self.buf[name][:, :0]


def decode_step_single(st: OpenChunkState):
    """One decode step, single sequence: rerun the chunk kernel over the open chunk."""
    b = st.buf
    full = st.fill == C
    o, fs = chunk(b["q"], b["k"], b["v"], b["g"], b["beta"],
                  initial_state=st.s_b, output_final_state=full)
    out = o[0, -1]
    st.roll_if_full(fs)
    return out


def section_a():
    print("--- A. single-sequence decode simulation vs full prefill ---", flush=True)
    fails = 0
    for prompt_len, gen in ((5, 130), (64, 70), (63, 66), (127, 80), (200, 100)):
        L = prompt_len + gen
        q, k, v, g, beta = make_seq(L, seed=prompt_len)
        o_full, _ = chunk(q, k, v, g, beta)  # what the TRAINER computes (one call, whole sequence)

        st = OpenChunkState.from_prefill(q[:, :prompt_len], k[:, :prompt_len], v[:, :prompt_len],
                                         g[:, :prompt_len], beta[:, :prompt_len])
        worst, bad = 0.0, 0
        for t in range(prompt_len, L):
            st.append(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g[:, t:t + 1], beta[:, t:t + 1])
            out = decode_step_single(st)
            d = maxdiff(out, o_full[0, t])
            if d != 0.0 or d != d:  # nonzero OR NaN
                bad += 1
                worst = d if (worst != worst or d != d or d > worst) else worst
        status = "BITWISE" if bad == 0 else f"DIFFERS {bad}/{gen} worst={worst:.3e}"
        print(f"  prompt={prompt_len:3d} gen={gen:3d}: {status}", flush=True)
        fails += bad
    return fails


def section_b():
    """Batched varlen decode: N requests, different open-chunk fills, decoded in ONE kernel call."""
    print("--- B. batched varlen decode (per-request bitwise vs its own full prefill) ---", flush=True)
    prompts = [5, 64, 63, 127, 200, 33, 128, 7]
    N = len(prompts)
    GEN = 70
    seqs = [make_seq(p + GEN, seed=100 + i) for i, p in enumerate(prompts)]
    fulls = [chunk(*s)[0] for s in seqs]
    states = [OpenChunkState.from_prefill(*(t[:, :p] for t in s)) for s, p in zip(seqs, prompts)]

    worst, bad, total = 0.0, 0, 0
    for step in range(GEN):
        for i, (s, p) in enumerate(zip(seqs, prompts)):
            t = p + step
            states[i].append(*(x[:, t:t + 1] for x in s))

        # pack every request's open chunk into one varlen call
        lens = [st.fill for st in states]
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
        cat = lambda name: torch.cat([st.buf[name] for st in states], dim=1)  # noqa: E731
        s0 = torch.cat([st.s_b for st in states], dim=0)  # [N, H, V, K]
        full_mask = [ln == C for ln in lens]
        o, fs = chunk(cat("q"), cat("k"), cat("v"), cat("g"), cat("beta"),
                      initial_state=s0, output_final_state=any(full_mask), cu_seqlens=cu)

        for i, st in enumerate(states):
            row = int(cu[i + 1]) - 1
            t = prompts[i] + step
            d = maxdiff(o[0, row], fulls[i][0, t])
            total += 1
            if d != 0.0 or d != d:
                bad += 1
                worst = d if (worst != worst or d != d or d > worst) else worst
            st.roll_if_full(fs[i : i + 1] if fs is not None else None)

    print(f"  N={N} requests x {GEN} steps, mixed open-chunk fills: "
          + ("BITWISE" if bad == 0 else f"DIFFERS {bad}/{total} worst={worst:.3e}"), flush=True)
    return bad


def section_c():
    print("--- C. cost of the recompute ---", flush=True)
    print(f"  open-chunk recompute processes 1..{C} rows per decoded token (mean {(C + 1) / 2:.1f}), "
          f"vs 1 row for the fused recurrent step.", flush=True)
    print(f"  => ~{(C + 1) / 2:.0f}x the token-rows on GDN layers only; C is tunable "
          f"(FLA_CHUNK_SIZE) and trades decode cost against training-kernel efficiency.", flush=True)


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    # Production pinning: without it the FLA autotuner selects a racy config for
    # chunk_scaled_dot_kkt (BK=64/w4/s>=2), and the kernel is neither deterministic nor
    # cross-sequence invariant -- section B then fails for reasons unrelated to decode.
    from skyrl.backends.skyrl_train.zerokl.gdn_batch_invariant import (
        pin_fla_autotune_configs,
        verify_gdn_batch_invariance,
    )

    pin_fla_autotune_configs()
    verify_gdn_batch_invariance()
    print("[gdn] kernels pinned + verified (deterministic, cross-seq invariant, prefix invariant)",
          flush=True)
    print(f"=== chunk-consistent GDN decode | H={HEADS} K={K_DIM} V={V_DIM} {DTYPE} C={C} ===",
          flush=True)
    fails = section_a() + section_b()
    section_c()
    if fails:
        print("\nRESULT: FAIL -- chunk-consistent decode is not bitwise.", flush=True)
        raise SystemExit(1)
    print("\nRESULT: chunk-consistent decode is BITWISE vs prefill, single AND batched varlen.",
          flush=True)


if __name__ == "__main__":
    main()
