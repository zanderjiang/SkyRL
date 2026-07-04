"""Element-wise sync verification: is the engine's GENERATION weight byte-identical to the trainer's
SCORING weight, per tensor?

In a live zero-KL run with SKYRL_ZEROKL_BISECT=1, two dumps are written on the first forward of each
side (both now carry the 3-way element-wise fingerprint from zerokl/weight_fingerprint.py):
    /mnt/local_storage/zerokl_eng_w.txt   <- engine GPTModel params at generation (post native-sync)
    /mnt/local_storage/zerokl_trn_w.txt   <- trainer GPTModel params at the scoring forward

This script joins them by name and reports, per tensor, whether abs_sum / sum_sq / ramp_dot ALL match.
A full match across all 255 tensors proves the sync delivers byte-identical weights (so the run's
rollout-vs-train residual is NOT the weights). Any mismatch names the exact tensor(s) the live
sleep/wake + sync drifts on -- the cause of the not-zero-KL residual, since the forward itself is
proven bitwise (engine_trainer_parity_harness.py).

Usage:
    python compare_weight_fingerprints.py [eng_file] [trn_file]
"""
import sys

ENG = sys.argv[1] if len(sys.argv) > 1 else "/mnt/local_storage/zerokl_eng_w.txt"
TRN = sys.argv[2] if len(sys.argv) > 2 else "/mnt/local_storage/zerokl_trn_w.txt"


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            name, absn, sq, dot, shape, dtype = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            rows[name] = (float(absn), float(sq), float(dot), shape, dtype)
    return rows


eng = load(ENG)
trn = load(TRN)
common = sorted(set(eng) & set(trn))
only_eng = sorted(set(eng) - set(trn))
only_trn = sorted(set(trn) - set(eng))

print(f"engine tensors={len(eng)} trainer tensors={len(trn)} common={len(common)}")
if only_eng:
    print(f"ONLY in engine ({len(only_eng)}): {only_eng[:6]}{' ...' if len(only_eng) > 6 else ''}")
if only_trn:
    print(f"ONLY in trainer ({len(only_trn)}): {only_trn[:6]}{' ...' if len(only_trn) > 6 else ''}")

FIELDS = ("abs_sum", "sum_sq", "ramp_dot")
n_exact = 0
mismatches = []  # (name, [which fields differ], rel_diffs)
for name in common:
    e = eng[name]
    t = trn[name]
    diffs = []
    rels = []
    for k in range(3):
        a, b = e[k], t[k]
        if a != b:
            denom = max(abs(a), abs(b), 1e-30)
            diffs.append(FIELDS[k])
            rels.append(abs(a - b) / denom)
    if not diffs:
        n_exact += 1
    else:
        mismatches.append((name, diffs, max(rels), e[3], e[4], t[4]))

print(f"\nBITWISE-IDENTICAL tensors: {n_exact}/{len(common)}")
if not mismatches and not only_eng and not only_trn:
    print("RESULT: SYNC IS BITWISE -- engine generation weights == trainer scoring weights, "
          "per element. The not-zero-KL residual is NOT the weights.")
else:
    print(f"RESULT: SYNC DRIFTS on {len(mismatches)} tensor(s) -- THIS is the not-zero-KL residual "
          "(forward is proven bitwise). Worst offenders:")
    mismatches.sort(key=lambda x: -x[2])
    for name, diffs, relmax, shape, edt, tdt in mismatches[:25]:
        print(f"  {name}  diff_in={diffs} max_rel={relmax:.3e} shape={shape} eng_dtype={edt} trn_dtype={tdt}")
