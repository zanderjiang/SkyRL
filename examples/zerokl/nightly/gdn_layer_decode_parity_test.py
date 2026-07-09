"""END-TO-END GDN layer parity: chunk-consistent decode == the training/prefill forward, BITWISE.

Unlike gdn_chunk_consistent_decode_sim.py (which exercises only the chunk kernel on synthetic
q/k/v/g/beta), this drives the FULL layer pipeline through the production class
`zerokl.gdn_chunk_consistent.ChunkConsistentGDN`, starting from raw pre-conv `mixed_qkv` and the
gating inputs a, b:

    conv (width-4 causal depthwise) -> split q/k/v -> L2norm(q), L2norm(k), GQA expand
      -> g = -exp(A_log)*softplus(a+dt_bias), beta = sigmoid(b)  -> chunk_gated_delta_rule

Reference ("what the trainer computes") = one prefill call over the whole prompt+response.
Under test = prefill(prompt) then one decode() per generated token, with N requests of DIFFERENT
prompt lengths decoded together so their open chunks have different fills.

Run: CUDA_VISIBLE_DEVICES=<gpu> uv run --isolated --extra zerokl \
       python examples/zerokl/nightly/gdn_layer_decode_parity_test.py
Exit 0 iff every generated token matches its reference row bitwise.
"""

import sys

import torch

sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")

from skyrl.backends.skyrl_train.zerokl.gdn_batch_invariant import (  # noqa: E402
    pin_fla_autotune_configs,
)
from skyrl.backends.skyrl_train.zerokl.gdn_chunk_consistent import ChunkConsistentGDN  # noqa: E402

# Qwen3.5-shaped GDN layer (scaled down in head count to keep the test quick)
NUM_K_HEADS, HEAD_K = 4, 128
NUM_V_HEADS, HEAD_V = 8, 128   # GQA 2x, exercises the repeat_interleave path
W = 4
CHUNK = 64
DTYPE = torch.bfloat16


def maxdiff(a, b):
    d = (a.float() - b.float()).abs().max()
    return float("nan") if torch.isnan(d) else float(d)


def build(num_slots, dev):
    qkv_dim = 2 * NUM_K_HEADS * HEAD_K + NUM_V_HEADS * HEAD_V
    torch.manual_seed(1234)
    return ChunkConsistentGDN(
        num_slots=num_slots,
        chunk_size=CHUNK,
        conv_weight=torch.randn(qkv_dim, W, dtype=DTYPE, device=dev) * 0.1,
        conv_bias=torch.randn(qkv_dim, dtype=DTYPE, device=dev) * 0.1,
        A_log=torch.randn(NUM_V_HEADS, device=dev) * 0.5,
        dt_bias=torch.randn(NUM_V_HEADS, device=dev) * 0.1,
        num_k_heads=NUM_K_HEADS, head_k_dim=HEAD_K,
        num_v_heads=NUM_V_HEADS, head_v_dim=HEAD_V,
        dtype=DTYPE, device=dev,
    ), qkv_dim


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    pin_fla_autotune_configs()
    dev = "cuda"

    prompts = [5, 64, 63, 127, 200, 33]
    GEN = 75
    N = len(prompts)
    layer, qkv_dim = build(N, dev)

    torch.manual_seed(7)
    seqs = []
    for i, p in enumerate(prompts):
        L = p + GEN
        seqs.append((
            torch.randn(L, qkv_dim, dtype=DTYPE, device=dev),
            torch.randn(L, NUM_V_HEADS, dtype=DTYPE, device=dev),
            torch.randn(L, NUM_V_HEADS, dtype=DTYPE, device=dev),
        ))

    # reference: the trainer's single full-sequence forward, per request
    refs = []
    for i in range(N):
        ref_layer, _ = build(1, dev)
        ref_layer.conv_weight, ref_layer.conv_bias = layer.conv_weight, layer.conv_bias
        ref_layer.A_log, ref_layer.dt_bias = layer.A_log, layer.dt_bias
        x, a, b = seqs[i]
        refs.append(ref_layer.prefill(0, x, a, b))

    # under test: prefill the prompt, then decode token by token, all requests together
    for i, p in enumerate(prompts):
        x, a, b = seqs[i]
        o_pref = layer.prefill(i, x[:p], a[:p], b[:p])
        d = maxdiff(o_pref, refs[i][:p])
        if d != 0.0:
            print(f"  prompt {i} prefill differs: {d:.3e}")

    slots = torch.arange(N, device=dev)
    bad, worst, total = 0, 0.0, 0
    for step in range(GEN):
        xs = torch.stack([seqs[i][0][prompts[i] + step] for i in range(N)])
        as_ = torch.stack([seqs[i][1][prompts[i] + step] for i in range(N)])
        bs = torch.stack([seqs[i][2][prompts[i] + step] for i in range(N)])
        out = layer.decode(slots, xs, as_, bs)
        for i in range(N):
            d = maxdiff(out[i], refs[i][prompts[i] + step])
            total += 1
            if d != 0.0 or d != d:
                bad += 1
                worst = d if (worst != worst or d != d or d > worst) else worst

    fills = layer.fill.tolist()
    print(f"requests={N} prompts={prompts} gen={GEN} (open-chunk fills at end: {fills})")
    if bad:
        print(f"\nRESULT: FAIL -- {bad}/{total} decoded tokens differ (worst {worst:.3e})")
        raise SystemExit(1)
    print(f"\nRESULT: {total}/{total} decoded tokens BITWISE == the full-sequence forward.")


if __name__ == "__main__":
    main()
