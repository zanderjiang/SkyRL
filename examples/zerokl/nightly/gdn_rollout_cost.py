"""What does chunk-consistent GDN decode cost in rollout throughput?

Decode re-runs the training chunk kernel over the OPEN chunk each step, so a decoded token costs
~(C+1)/2 token-rows of GDN work instead of 1 (C = FLA_CHUNK_SIZE = 64), on the GDN layers only --
which are 3 of every 4 Qwen3.5 layers, but are cheap linear attention next to the MLPs.

THREE ARMS, because a two-arm A/B would blame chunk-consistent decode for the whole zero-KL stack:

  stock : vLLM as shipped -- fused recurrent decode, no batch invariance, default attention backend.
          Fast, and NOT bitwise (2.52% of tokens exact).
  bi    : batch-invariant kernels + the num_splits=1 CUSTOM varlen backend + Triton matmuls, but
          vLLM's STOCK GDN decode. Still not bitwise (that is the whole point of this workstream);
          it isolates what the rest of the zero-KL stack costs on its own.
  cc    : bi + chunk-consistent GDN decode. BITWISE (65536/65536 exact).

So `bi / cc` is the price of chunk-consistent decode, and `stock / cc` is the price of bitwise
zero-KL end to end. Report both; do not trade correctness for speed silently. If the cost is
unacceptable, C is the knob -- but it must match on the trainer and the engine, because it defines
the chunk grid the recurrent state is pinned to.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=examples/zerokl/nightly/_torchvision_stub \
      HF_HOME=/mnt/local_storage/hf uv run --isolated --extra zerokl \
      python examples/zerokl/nightly/gdn_rollout_cost.py
"""

import os
import subprocess
import sys
import time

MODEL = os.environ.get("GDN_MODEL", "Qwen/Qwen3.5-0.8B")
NSEQ = int(os.environ.get("GDN_NSEQ", "16"))
NTOK = int(os.environ.get("GDN_NTOK", "512"))
PROMPTS = [
    "Explain, step by step, how a compiler turns source code into machine code.",
    "Write a detailed proof that the square root of two is irrational.",
    "Describe the causes and consequences of the Bronze Age collapse.",
    "Solve: a train travels 60 mph for 2.5 hours, then 40 mph for 1.5 hours. Show all work.",
]


ARMS = ("stock", "bi", "cc")


def run_one(arm: str):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if arm != "stock":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm import LLM, SamplingParams

    if arm != "stock":
        sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")
        from skyrl.backends.skyrl_train.zerokl.gdn_batch_invariant import (
            pin_fla_autotune_configs, pin_gdn_rmsnorm_rows_per_block)
        from skyrl.backends.skyrl_train.zerokl.moe_batch_invariant import _install_moe_matmul_invariance
        from skyrl.backends.skyrl_train.zerokl.varlen_backend import register_varlen_custom_backend

        register_varlen_custom_backend()
        _install_moe_matmul_invariance()
        pin_fla_autotune_configs()
        pin_gdn_rmsnorm_rows_per_block()
        # Lift vLLM's GDN batch-invariance veto in BOTH arms, so `bi` differs from `cc` only in the
        # decode kernel -- not in whether the engine would boot.
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

        GDNAttentionBackend.supports_batch_invariance = classmethod(lambda cls: True)
        if arm == "cc":
            from skyrl.backends.skyrl_train.zerokl import gdn_engine_patch as gep

            # force=True: this arm deliberately does not set SKYRL_ZEROKL_GDN (the `bi` arm must not
            # inherit it), and without force the install is a silent no-op -- which reads as
            # "chunk-consistent decode is free".
            gep.install_gdn_engine_patch(force=True)

    llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True, max_num_seqs=NSEQ,
              gpu_memory_utilization=0.55, max_model_len=NTOK + 512, enable_prefix_caching=False,
              enable_chunked_prefill=False, trust_remote_code=True, seed=0,
              **({"attention_backend": "CUSTOM"} if arm != "stock" else {}),
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    pids = [tok(PROMPTS[i % len(PROMPTS)], add_special_tokens=False).input_ids for i in range(NSEQ)]
    sp = SamplingParams(temperature=1.0, top_p=1.0, seed=0, max_tokens=NTOK, ignore_eos=True)

    llm.generate([{"prompt_token_ids": pids[0]}], SamplingParams(max_tokens=8, ignore_eos=True))  # warm
    t0 = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": p} for p in pids], sp)
    dt = time.perf_counter() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)

    if arm == "cc":
        from skyrl.backends.skyrl_train.zerokl.gdn_engine_patch import forward_core_call_count

        if forward_core_call_count() == 0:
            raise SystemExit("arm=cc never called the patched _forward_core; this timing is arm=bi")
    print(f"RESULT arm={arm} tokens={n} seconds={dt:.2f} gen_tok_per_s={n / dt:.1f}", flush=True)


def main():
    if "--child" in sys.argv:
        run_one(os.environ["GDN_COST_ARM"])
        return
    results = {}
    for arm in ARMS:
        env = dict(os.environ, GDN_COST_ARM=arm)
        env.pop("VLLM_BATCH_INVARIANT", None)
        env.pop("SKYRL_ZEROKL_GDN", None)
        print(f"\n{'=' * 80}\n### arm={arm}\n{'=' * 80}", flush=True)
        out = subprocess.run([sys.executable, __file__, "--child"], env=env,
                             capture_output=True, text=True)
        sys.stdout.write(out.stdout[-4000:])
        if out.returncode != 0:
            sys.stderr.write(out.stderr[-4000:])
            raise SystemExit(out.returncode)
        for line in out.stdout.splitlines():
            if line.startswith("RESULT arm="):
                results[arm] = float(line.split("gen_tok_per_s=")[1])

    if len(results) == 3:
        stock, bi, cc = results["stock"], results["bi"], results["cc"]
        print(f"\n{'=' * 80}\nROLLOUT COST (model={MODEL}, {NSEQ} seqs x {NTOK} tok)\n{'=' * 80}")
        print(f"  stock vLLM                       (NOT bitwise): {stock:8.1f} gen tok/s")
        print(f"  + batch-invariant stack          (NOT bitwise): {bi:8.1f} gen tok/s   "
              f"({stock / bi:.2f}x vs stock)")
        print(f"  + chunk-consistent GDN decode    (BITWISE)    : {cc:8.1f} gen tok/s   "
              f"({bi / cc:.2f}x vs bi)")
        print(f"\n  price of chunk-consistent decode alone : {bi / cc:.2f}x")
        print(f"  price of bitwise zero-KL end to end   : {stock / cc:.2f}x")


if __name__ == "__main__":
    main()
