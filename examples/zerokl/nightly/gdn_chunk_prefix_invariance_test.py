"""DECISIVE experiment for GDN zero-KL: is the chunk kernel PREFIX-INVARIANT, bitwise?

The chunk-consistent-decode design rests on exactly one property:

    chunk_gated_delta_rule(x[0 : t+1], initial_state=S)[t]
      ==  (bitwise)
    chunk_gated_delta_rule(x[0 : L],   initial_state=S)[t]        for every t < L

i.e. a token's output does not depend on how many tokens come AFTER it in the chunk. GDN is
causal, so this is true in exact arithmetic; whether the Triton kernel preserves it is an
implementation question. If it holds, decode can be made bitwise-identical to prefill/training by
(a) keeping the recurrent state only at chunk boundaries and (b) re-running the SAME chunk kernel
over the open chunk's prefix at every decode step. If it fails, the whole approach is dead and no
amount of plumbing saves it.

The math says it should hold: every intra-chunk op is causally row-local -- solve_tril is a forward
substitution (row t from rows <= t), chunk_scaled_dot_kkt / chunk_fwd_o use tril masks, and
chunk_local_cumsum is an inclusive scan. The one op that reads the whole chunk is the inter-chunk
state advance (chunk_delta_h uses g_last), and that only ever runs on a FULL chunk.

Run (one free GPU, zerokl nightly venv):
    CUDA_VISIBLE_DEVICES=<gpu> uv run --isolated --extra zerokl \
      python examples/zerokl/nightly/gdn_chunk_prefix_invariance_test.py

Env knobs: GDN_SEQLEN (256), GDN_HEADS (32), GDN_K (128), GDN_V (128), GDN_DTYPE (bfloat16).
Exit code 0 iff every probe is bitwise (max abs diff == 0.0).
"""

import os

import torch

SEQLEN = int(os.environ.get("GDN_SEQLEN", "256"))
HEADS = int(os.environ.get("GDN_HEADS", "32"))
K_DIM = int(os.environ.get("GDN_K", "128"))
V_DIM = int(os.environ.get("GDN_V", "128"))
DTYPE = getattr(torch, os.environ.get("GDN_DTYPE", "bfloat16"))
CHUNK = 64  # FLA_CHUNK_SIZE


def make_inputs(seed=0):
    """Realistic GDN inputs: g is log-space decay (<=0), beta in (0,1)."""
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(1, SEQLEN, HEADS, K_DIM, dtype=DTYPE, device=dev)
    k = torch.randn(1, SEQLEN, HEADS, K_DIM, dtype=DTYPE, device=dev)
    v = torch.randn(1, SEQLEN, HEADS, V_DIM, dtype=DTYPE, device=dev)
    # g: log of a per-token forget gate in (0,1) -> strictly negative, like -softplus(.)*A
    g = -torch.nn.functional.softplus(torch.randn(1, SEQLEN, HEADS, device=dev)).float()
    beta = torch.rand(1, SEQLEN, HEADS, dtype=DTYPE, device=dev).sigmoid()
    return q, k, v, g, beta


def run_chunk(q, k, v, g, beta, initial_state=None, output_final_state=False):
    """Returns (o, final_state); final_state is None unless output_final_state."""
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

    o, final_state = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
    )
    return o, final_state


def probe_positions():
    """Cover: chunk interiors, both sides of every boundary, first/last token."""
    pts = {0, 1, 2, CHUNK - 2, CHUNK - 1, CHUNK, CHUNK + 1, SEQLEN - 1}
    pts |= {CHUNK * i + d for i in range(1, SEQLEN // CHUNK) for d in (-1, 0, 1, 7)}
    pts |= {13, 37, 63, 64, 65, 100, 127, 128, 129}
    return sorted(p for p in pts if 0 <= p < SEQLEN)


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    print(f"=== GDN chunk PREFIX-INVARIANCE | torch {torch.__version__} | L={SEQLEN} "
          f"H={HEADS} K={K_DIM} V={V_DIM} dtype={DTYPE} chunk={CHUNK} ===", flush=True)

    q, k, v, g, beta = make_inputs()

    # Two regimes: fresh state (start of sequence) and a carried-in state (mid-sequence chunk,
    # which is what decode always sees).
    regimes = {
        "initial_state=None": None,
        "initial_state=random": torch.randn(1, HEADS, V_DIM, K_DIM, dtype=torch.float32, device="cuda"),
    }

    failures = []
    for name, s0 in regimes.items():
        o_full, _ = run_chunk(q, k, v, g, beta, initial_state=s0)
        worst = 0.0
        worst_t = -1
        nonzero = 0
        probes = probe_positions()
        for t in probes:
            o_pref, _ = run_chunk(
                q[:, : t + 1], k[:, : t + 1], v[:, : t + 1], g[:, : t + 1], beta[:, : t + 1],
                initial_state=s0,
            )
            d = float((o_pref[0, t].float() - o_full[0, t].float()).abs().max())
            if d != 0.0:
                nonzero += 1
                if d > worst:
                    worst, worst_t = d, t
        status = "BITWISE" if nonzero == 0 else f"DIFFERS on {nonzero}/{len(probes)} probes"
        print(f"[{name}] prefix-vs-full row t: {status}"
              + ("" if nonzero == 0 else f"  worst t={worst_t} max={worst:.6e}"), flush=True)
        if nonzero:
            failures.append(name)

    # The state advance must also be exact: state after a FULL chunk, computed alone vs as the
    # first chunk of a longer sequence. (Decode only ever advances the state on full chunks.)
    s0 = regimes["initial_state=random"]
    _, st_alone = run_chunk(q[:, :CHUNK], k[:, :CHUNK], v[:, :CHUNK], g[:, :CHUNK], beta[:, :CHUNK],
                            initial_state=s0, output_final_state=True)
    _, st_long = run_chunk(q[:, : 2 * CHUNK], k[:, : 2 * CHUNK], v[:, : 2 * CHUNK],
                           g[:, : 2 * CHUNK], beta[:, : 2 * CHUNK],
                           initial_state=s0, output_final_state=True)
    # st_long is the state after TWO chunks; recompute chunk 2 from st_alone and compare.
    _, st_chained = run_chunk(q[:, CHUNK : 2 * CHUNK], k[:, CHUNK : 2 * CHUNK], v[:, CHUNK : 2 * CHUNK],
                              g[:, CHUNK : 2 * CHUNK], beta[:, CHUNK : 2 * CHUNK],
                              initial_state=st_alone, output_final_state=True)
    d_state = float((st_chained.float() - st_long.float()).abs().max())
    print(f"[state advance] chunk-by-chunk chaining vs one 2-chunk call: max={d_state:.6e} "
          f"{'BITWISE' if d_state == 0.0 else 'DIFFERS'}", flush=True)
    if d_state != 0.0:
        failures.append("state-chaining")

    if failures:
        print(f"\nRESULT: FAIL -- chunk kernel is NOT prefix-invariant ({', '.join(failures)}).\n"
              "Chunk-consistent decode cannot be bitwise with this kernel as-is.", flush=True)
        raise SystemExit(1)
    print("\nRESULT: PREFIX-INVARIANT + state chaining exact. Chunk-consistent decode is viable.",
          flush=True)


if __name__ == "__main__":
    main()
