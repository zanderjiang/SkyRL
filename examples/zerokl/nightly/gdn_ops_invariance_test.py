"""Invariance of the shared GDN ops (zerokl/gdn_ops.py) that decode-vs-prefill parity rests on.

  A. gdn_causal_conv is prefix invariant:  conv(x[:t+1])[t] == conv(x)[t]
  B. gdn_causal_conv resumes exactly from a boundary state:
        conv(x[B:j], init=state_after_B)[-1] == conv(x)[j-1]
  C. gdn_causal_conv is invariant to unrelated tokens (no batch/sequence coupling)
  D. gdn_chunk (pinned) is deterministic + cross-sequence invariant + prefix invariant

Run: CUDA_VISIBLE_DEVICES=<gpu> uv run --isolated --extra zerokl \
       python examples/zerokl/nightly/gdn_ops_invariance_test.py
Exit 0 iff every check is bitwise.
"""

import sys

import torch

sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")

from skyrl.backends.skyrl_train.zerokl.gdn_batch_invariant import (  # noqa: E402
    pin_fla_autotune_configs,
    verify_gdn_batch_invariance,
)
from skyrl.backends.skyrl_train.zerokl.gdn_ops import gdn_causal_conv  # noqa: E402

D, W, L = 512, 4, 200
CH = 64


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(L, D, dtype=torch.bfloat16, device=dev)
    w = torch.randn(D, W, dtype=torch.bfloat16, device=dev)
    bias = torch.randn(D, dtype=torch.bfloat16, device=dev)

    fails = []
    y_full, st_full = gdn_causal_conv(x, w, bias, return_final_state=True)

    # A. prefix invariance
    bad = [t for t in (0, 1, 3, 10, 63, 64, 65, 127, 199)
           if not torch.equal(gdn_causal_conv(x[: t + 1], w, bias)[t], y_full[t])]
    print(f"A. conv prefix invariance: {'BITWISE' if not bad else f'FAIL at {bad}'}")
    if bad:
        fails.append("conv-prefix")

    # B. resume from a chunk-boundary state (what decode does every step)
    B = 128
    _, st_B = gdn_causal_conv(x[:B], w, bias, return_final_state=True)
    bad2 = [j for j in range(B + 1, B + 20)
            if not torch.equal(gdn_causal_conv(x[B:j], w, bias, initial_state=st_B)[-1], y_full[j - 1])]
    print(f"B. conv open-chunk resume: {'BITWISE' if not bad2 else f'FAIL at {bad2[:5]}'}")
    if bad2:
        fails.append("conv-resume")

    # C. no coupling to unrelated tokens: a different tail must not change earlier rows
    x2 = x.clone()
    x2[100:] = torch.randn(L - 100, D, dtype=torch.bfloat16, device=dev)
    y2 = gdn_causal_conv(x2, w, bias)
    same_prefix = torch.equal(y2[:100], y_full[:100])
    print(f"C. conv independence of later tokens: {'BITWISE' if same_prefix else 'FAIL'}")
    if not same_prefix:
        fails.append("conv-coupling")

    # final state must be exactly the last W-1 raw inputs
    if not torch.equal(st_full, x[-(W - 1) :].transpose(0, 1).contiguous()):
        fails.append("conv-final-state")
        print("   conv final_state mismatch")

    # D. chunk kernel properties (the production verifier)
    pin_fla_autotune_configs()
    verify_gdn_batch_invariance()
    print("D. gdn_chunk pinned: deterministic + cross-seq invariant + prefix invariant: BITWISE")

    if fails:
        print(f"\nRESULT: FAIL ({', '.join(fails)})")
        raise SystemExit(1)
    print("\nRESULT: all shared GDN ops are bitwise invariant.")


if __name__ == "__main__":
    main()
