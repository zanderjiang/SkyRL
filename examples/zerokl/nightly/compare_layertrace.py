"""Localize the distributed-machinery divergence: diff the LIVE fbf/Float16Module forward intermediates
(trace_live_trainer.pt, dumped inside the real trainer actor) against the CLEAN single-process bare
forward intermediates (trace_clean_trainer.pt) on the SAME micro-batch-0 tokens, in forward order.

The first module whose output diverges (while its inputs matched) is the op the distributed training
forward introduces the ~0.014 at. Both are the same Megatron GPTModel (same weights), so modules align
by name; the only difference is the execution path (fbf/Float16Module/DDP vs bare).
"""
import torch

live = torch.load("/mnt/local_storage/trace_live_trainer.pt", map_location="cpu")   # {name: tensor}
cln = torch.load("/mnt/local_storage/trace_clean_trainer.pt", map_location="cpu")    # {store,order,P,na,L}
clean = cln["store"]; order = cln["order"]; P = cln["P"]; na = cln["na"]; Lc = cln["L"]
print(f"live modules={len(live)} clean modules={len(clean)} P={P} na={na} L={Lc}")


def to_LH(t, L):
    shp = tuple(t.shape)
    if L not in shp:
        return None
    return t.movedim(shp.index(L), 0).reshape(L, -1)


common = [n for n in sorted(order, key=lambda x: order[x]) if n in live and n in clean]
print(f"common modules (forward order): {len(common)}\n")
print(f"{'idx':>4} {'module':<50} {'live_shape':<18} {'max':>11} {'mean':>11}")
rows = []
first_nonzero = None
for n in common:
    tl, tc = live[n], clean[n]
    # find a token dim present in both (prefer L=clean L; else the largest shared dim)
    Ls = [d for d in tl.shape if d in tc.shape and d > 1]
    if not Ls:
        continue
    L = Lc if Lc in Ls else max(Ls)
    a, b = to_LH(tl, L), to_LH(tc, L)
    if a is None or b is None or a.shape != b.shape:
        continue
    # compare over the response region if L matches the full seq, else whole
    if L == Lc and P < L:
        a, b = a[P:], b[P:]
    dd = (a.float() - b.float()).abs()
    mx, mn = float(dd.max()), float(dd.mean())
    rows.append((order[n], n, tuple(tl.shape), mx, mn))
    if first_nonzero is None and mn > 0:
        first_nonzero = n

for idx, n, shp, mx, mn in rows:
    mark = "  <== FIRST DIVERGENCE" if n == first_nonzero else ""
    print(f"{idx:>4} {n:<50} {str(shp):<18} {mx:>11.3e} {mn:>11.3e}{mark}")

print(f"\nFIRST module to diverge (live fbf vs clean bare): {first_nonzero}")

# ---- decisive: are the core_attention INPUTS (post-RoPE q/k/v) identical live vs clean? ----
print("\n=== core_attention INPUTS (post-RoPE q/k/v) live vs clean -- layer 0 ===")
base = "decoder.layers.0.self_attention.core_attention"
for j, nm in enumerate(("q", "k", "v")):
    key = f"{base}::IN{j}"
    if key in live and key in clean:
        a, b = live[key], clean[key]
        if a.shape == b.shape:
            dd = (a.float() - b.float()).abs()
            print(f"  input {nm} {tuple(a.shape)}: max={float(dd.max()):.3e} mean={float(dd.mean()):.3e} "
                  f"{'IDENTICAL' if float(dd.max())==0 else 'DIFFER'}")
        else:
            print(f"  input {nm}: SHAPE MISMATCH live={tuple(a.shape)} clean={tuple(b.shape)}")
    else:
        print(f"  input {nm}: missing (live={key in live} clean={key in clean})")
if f"{base}" in live and f"{base}" in clean:
    a = to_LH(live[base], Lc); b = to_LH(clean[base], Lc)
    if a is not None and b is not None and a.shape == b.shape:
        dd = (a - b).abs()
        print(f"  OUTPUT: max={float(dd.max()):.3e} mean={float(dd.mean()):.3e}")
print("  => if inputs IDENTICAL but output DIFFERS: kernel is runtime-dependent on THESE real inputs.")
print("  => if inputs DIFFER: divergence is upstream (RoPE/split), not the kernel.")
# residual stream after each decoder layer
print("\n=== residual stream after each decoder layer (live vs clean) ===")
for idx, n, shp, mx, mn in rows:
    if n.startswith("decoder.layers.") and n.count(".") == 2:
        print(f"  {n:<26} max={mx:.3e} mean={mn:.3e}")
