"""Measure decode-vs-prefill logprob divergence of Qwen3.5's GatedDeltaNet layers. MEASUREMENT ONLY.

Phase 2. Plain native vLLM -- NOT the zero-KL GPTModel wrapper, no Megatron, no Ray, one GPU. The
point is to quantify how far a linear-attention (GatedDeltaNet) model sits from decode == prefill
parity, so the separate GDN workstream knows what it is chasing. Nothing here is a fix.

Method: generate N sequences of L tokens at temperature 1.0 with per-token logprobs, then feed each
completed sequence back through the *same* engine as a prompt and read ``prompt_logprobs`` for the
exact generated ids. Decode saw a recurrent state advanced one token at a time; prefill recomputes
it chunk-wise over the whole sequence. Their difference is the quantity of interest.

Softmax attention with the ``num_splits=1`` varlen kernel gives max == 0.0 here. Any nonzero spread
is the linear-attention path (plus, in the default run, whatever batch-variance the non-invariant
kernels contribute -- which is why we also report a ``VLLM_BATCH_INVARIANT=1`` run: it isolates the
GDN chunk-vs-recurrent gap from ordinary batch-variance).

Run both modes (spawns one subprocess per mode so the env is set before vLLM imports):
    CUDA_VISIBLE_DEVICES=<gpu> uv run --isolated --extra zerokl \
      python examples/zerokl/nightly/gdn_decode_prefill_divergence.py --both \
      > /mnt/local_storage/logs/gdn_divergence.log 2>&1

Env knobs: GDN_MODEL (Qwen/Qwen3.5-0.8B), GDN_NSEQ (32), GDN_NTOK (2048), GDN_BATCH_INVARIANT (0/1).
"""

import os
import statistics
import subprocess
import sys

MODEL = os.environ.get("GDN_MODEL", "Qwen/Qwen3.5-0.8B")
NSEQ = int(os.environ.get("GDN_NSEQ", "32"))
NTOK = int(os.environ.get("GDN_NTOK", "2048"))
# Prompts are deliberately varied in length and topic: GDN's recurrent state is history-dependent,
# so a single repeated prompt would under-sample the divergence.
PROMPTS = [
    "Explain, step by step, how a compiler turns source code into machine code.",
    "Write a detailed proof that the square root of two is irrational.",
    "Describe the causes and consequences of the Bronze Age collapse.",
    "Solve: a train travels 60 mph for 2.5 hours, then 40 mph for 1.5 hours. Show all work.",
    "Summarize how photosynthesis converts light into chemical energy, in depth.",
    "Derive the closed form of the Fibonacci sequence and verify it for n = 10.",
    "Compare and contrast TCP and UDP, including when each is appropriate.",
    "Explain the Monty Hall problem and why switching wins two thirds of the time.",
]


def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def run_one(batch_invariant: bool):
    import vllm.envs as vllm_envs
    from vllm import LLM, SamplingParams

    print(
        f"=== GDN decode-vs-prefill | model={MODEL} nseq={NSEQ} ntok={NTOK} "
        f"VLLM_BATCH_INVARIANT={vllm_envs.VLLM_BATCH_INVARIANT} ===",
        flush=True,
    )
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.80,
        max_model_len=NTOK + 512,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        trust_remote_code=True,
        seed=0,
        # Text-only measurement: Qwen3_5ForConditionalGeneration registers as multimodal, and
        # vLLM's startup profiling would otherwise run the vision processor on a dummy image
        # (which also needs torchvision, absent from the zerokl env).
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tok = llm.get_tokenizer()

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(NSEQ)]
    prompt_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]

    gen = llm.generate(
        [{"prompt_token_ids": p} for p in prompt_ids],
        SamplingParams(temperature=1.0, top_p=1.0, seed=0, max_tokens=NTOK, logprobs=0, ignore_eos=True),
    )

    # Rescore each completed sequence by prefill, reading prompt_logprobs at the generated positions.
    rescore_inputs, decode_lps, gen_lens = [], [], []
    for pids, out in zip(prompt_ids, gen):
        comp = out.outputs[0]
        ids = list(comp.token_ids)
        decode_lps.append([comp.logprobs[i][ids[i]].logprob for i in range(len(ids))])
        gen_lens.append(len(ids))
        rescore_inputs.append({"prompt_token_ids": list(pids) + ids})

    rescored = llm.generate(rescore_inputs, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))

    # abs diff per (sequence, generated position)
    per_position = [[] for _ in range(max(gen_lens))]
    all_diffs = []
    for pids, out2, dlps in zip(prompt_ids, rescored, decode_lps):
        full = out2.prompt_token_ids
        plps = out2.prompt_logprobs
        offset = len(pids)
        for i, dlp in enumerate(dlps):
            pos = offset + i
            plp = plps[pos][full[pos]].logprob
            d = abs(dlp - plp)
            all_diffs.append(d)
            per_position[i].append(d)

    srt = sorted(all_diffs)
    exact0 = sum(1 for d in all_diffs if d == 0.0)
    print(
        f"\ntokens compared : {len(all_diffs)}\n"
        f"exact 0.0       : {exact0}/{len(all_diffs)} ({100.0 * exact0 / len(all_diffs):.2f}%)\n"
        f"mean |diff|     : {statistics.fmean(all_diffs):.6e}\n"
        f"P50 |diff|      : {percentile(srt, 0.50):.6e}\n"
        f"P99 |diff|      : {percentile(srt, 0.99):.6e}\n"
        f"max |diff|      : {srt[-1]:.6e}",
        flush=True,
    )

    print("\ndiff vs position (mean / max over sequences, bucketed by generated-token index):", flush=True)
    bucket = max(1, max(gen_lens) // 16)
    for start in range(0, max(gen_lens), bucket):
        vals = [d for row in per_position[start : start + bucket] for d in row]
        if vals:
            print(
                f"  pos {start:5d}-{min(start + bucket, max(gen_lens)) - 1:5d}: "
                f"mean={statistics.fmean(vals):.6e}  max={max(vals):.6e}  n={len(vals)}",
                flush=True,
            )


def main():
    if "--both" in sys.argv:
        for bi in ("0", "1"):
            env = dict(os.environ, VLLM_BATCH_INVARIANT=bi, GDN_BATCH_INVARIANT=bi)
            print(f"\n{'=' * 88}\n### VLLM_BATCH_INVARIANT={bi}\n{'=' * 88}", flush=True)
            rc = subprocess.run([sys.executable, __file__], env=env).returncode
            if rc != 0:
                raise SystemExit(rc)
        return
    run_one(os.environ.get("VLLM_BATCH_INVARIANT") == "1")


if __name__ == "__main__":
    main()
