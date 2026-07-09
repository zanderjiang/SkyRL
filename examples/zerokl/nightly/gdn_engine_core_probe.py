"""Split layer 0's decode-vs-prefill divergence into "before GDN" vs "inside GDN".

The layer bisect says layer 0 (a GatedDeltaNet layer) is the first to diverge. Two candidates:

  A. its INPUT already differs -- ``mixed_qkv`` comes from the ``in_proj`` GEMM, run at M=1 during
     decode and M=T during prefill. A GEMM that is not M-invariant makes every downstream op differ,
     and no amount of chunk-consistent decode can fix it.
  B. its OUTPUT differs given identical input -- then chunk-consistent decode really is mis-wired.

So capture both sides of ``_forward_core`` for one layer, key them by absolute position, and compare.
Layer-level math is already proven bitwise (gdn_layer_decode_parity_test, 450/450), so A is the
hypothesis with prior support; this decides it with a measurement.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=examples/zerokl/nightly/_torchvision_stub \
      HF_HOME=/mnt/local_storage/hf SKYRL_ZEROKL_GDN=1 \
      uv run --isolated --extra zerokl python examples/zerokl/nightly/gdn_engine_core_probe.py
"""

import os
import sys

import torch

sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")

MODEL = os.environ.get("GDN_MODEL", "Qwen/Qwen3.5-0.8B")
NGEN = int(os.environ.get("GDN_NGEN", "200"))
LAYER = os.environ.get("GDN_PROBE_LAYER", "language_model.model.layers.0.linear_attn")
PROMPT = "Explain, step by step, how a compiler turns source code into machine code."


def maxdiff(a, b):
    d = (a.float() - b.float()).abs().max()
    return float("nan") if torch.isnan(d) else float(d)


def main():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm import LLM, SamplingParams

    from skyrl.backends.skyrl_train.zerokl import gdn_engine_patch as gep
    from skyrl.backends.skyrl_train.zerokl.moe_batch_invariant import _install_moe_matmul_invariance
    from skyrl.backends.skyrl_train.zerokl.varlen_backend import register_varlen_custom_backend

    gep.install_gdn_engine_patch()
    register_varlen_custom_backend()
    _install_moe_matmul_invariance()

    rec = {"phase": "prefill", "x": {}, "out": {}, "pos": 0}
    inner = gep._zerokl_forward_core

    def probe(self, mixed_qkv, b, a, core_attn_out):
        if self.prefix != LAYER:
            return inner(self, mixed_qkv, b, a, core_attn_out)
        inner(self, mixed_qkv, b, a, core_attn_out)
        n = mixed_qkv.shape[0]
        ph, base = rec["phase"], rec["pos"]
        for i in range(n):
            rec["x"].setdefault(ph, {})[base + i] = mixed_qkv[i].detach().float().cpu().clone()
            rec["out"].setdefault(ph, {})[base + i] = core_attn_out[i].detach().float().cpu().clone()
        rec["pos"] = 0 if ph == "prefill" else base + n

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

    QwenGatedDeltaNetAttention._forward_core = probe

    llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True, max_num_seqs=1,
              gpu_memory_utilization=0.55, max_model_len=NGEN + 512, enable_prefix_caching=False,
              enable_chunked_prefill=False, trust_remote_code=True, seed=0,
              attention_backend="CUSTOM", limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    pids = tok(PROMPT, add_special_tokens=False).input_ids
    P = len(pids)

    # phase 1: prompt prefill (positions 0..P-1) then NGEN decode steps (positions P..P+NGEN-1)
    rec["phase"], rec["pos"] = "decode", 0
    out = llm.generate([{"prompt_token_ids": pids}],
                       SamplingParams(temperature=1.0, top_p=1.0, seed=0, max_tokens=NGEN,
                                      logprobs=0, ignore_eos=True))[0]
    gen_ids = list(out.outputs[0].token_ids)

    # phase 2: one prefill over prompt+generated (positions 0..P+NGEN-1)
    rec["phase"], rec["pos"] = "prefill", 0
    llm.generate([{"prompt_token_ids": pids + gen_ids}],
                 SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))

    C = 64
    dec_x, dec_o = rec["x"]["decode"], rec["out"]["decode"]
    pre_x, pre_o = rec["x"]["prefill"], rec["out"]["prefill"]

    print(f"\nlayer={LAYER}  prompt={P} tok  generated={len(gen_ids)} tok  chunk={C}")
    print(f"open-chunk fill after prompt prefill = {P % C}; chunk boundaries at abs pos "
          f"{[p for p in range(C, P + len(gen_ids), C)]}\n")
    print(f"{'abs pos':>8} {'dec step':>9} {'fill':>5} {'|dx| (GDN input)':>18} {'|do| (GDN output)':>19}")

    first_x = first_o = None
    for t in range(len(gen_ids)):
        p = P + t
        if p not in dec_x or p not in pre_x:
            continue
        dx, do = maxdiff(dec_x[p], pre_x[p]), maxdiff(dec_o[p], pre_o[p])
        if dx != 0.0 and first_x is None:
            first_x = (p, t, dx)
        if do != 0.0 and first_o is None:
            first_o = (p, t, do)
        if t < 4 or (first_x and t <= first_x[1] + 2) or (first_o and t <= first_o[1] + 2):
            print(f"{p:>8} {t:>9} {(p % C) + 1:>5} {dx:>18.3e} {do:>19.3e}")

    print()
    if first_x:
        p, t, d = first_x
        print(f"GDN *INPUT* (mixed_qkv) first differs at abs pos {p} (decode step {t}), max {d:.3e}")
        print("  -> the in_proj GEMM is not M-invariant. This is UPSTREAM of chunk-consistent decode.")
    else:
        print("GDN input (mixed_qkv) is BITWISE identical at every generated position.")
    if first_o:
        p, t, d = first_o
        print(f"GDN *OUTPUT* first differs at abs pos {p} (decode step {t}), max {d:.3e}")
    else:
        print("GDN output is BITWISE identical at every generated position.")

    if first_x and first_o and first_x[0] == first_o[0]:
        print("\nRESULT: input and output diverge at the SAME position -> the GDN core is faithful; "
              "the bug is the projection GEMM feeding it.")
    elif first_o and not first_x:
        print("\nRESULT: identical input, different output -> chunk-consistent decode IS mis-wired.")


if __name__ == "__main__":
    main()
