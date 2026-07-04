"""Does the zero-KL engine give BITWISE logprobs under CONTINUOUS BATCHING?

The single-sequence parity test is 256/256 bitwise, but the live RL loop generates many sequences in
one batch (continuous batching). This checks the batched case: generate N sequences together, capture
each sampled token's decode logprob, then RE-SCORE each sequence ALONE (batch of 1) via prompt_logprobs.
If decode(batched) == rescore(alone) bitwise, the engine is batch-invariant and the live residual is
elsewhere; if they diverge on some tokens, the residual is the engine's batched batch-variance.
"""
import os

os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

N = int(os.environ.get("PARITY_NSEQ", "8"))      # sequences generated together (continuous batching)
NTOK = int(os.environ.get("PARITY_NTOK", "128"))
MODEL = os.environ.get("ZEROKL_MODEL", "/mnt/local_storage/models/MiMo-7B-RL")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
import skyrl.backends.skyrl_train.zerokl.varlen_backend as varlen_backend  # noqa: E402,F401
from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import register_gptmodel_to_vllm, VLLM_MODEL_NAME  # noqa: E402
from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env  # noqa: E402

apply_vllm_zerokl_env()
register_gptmodel_to_vllm()
_GMU = float(os.environ.get("ZK_ENGINE_GPU_MEM_UTIL", "0.55"))
_MAXSEQ = int(os.environ.get("ZK_MAX_NUM_SEQS", str(max(16, N + 4))))
_MAXLEN = int(os.environ.get("ZK_MAX_MODEL_LEN", "2048"))
llm = LLM(model=MODEL, hf_overrides={"architectures": [VLLM_MODEL_NAME]}, attention_backend="CUSTOM",
          dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=_GMU, max_model_len=_MAXLEN,
          max_num_seqs=_MAXSEQ,
          enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True)
tok = llm.get_tokenizer()

# N distinct prompts of distinct lengths, generated TOGETHER (one llm.generate call -> continuous batch).
prompts = [f"Question {i}: explain step by step why {i*7+3} plus {i*5+1} matters in arithmetic and life."
           for i in range(N)]
pids = [tok(p, add_special_tokens=False).input_ids[:32 + i] for i, p in enumerate(prompts)]
gen = llm.generate([{"prompt_token_ids": p} for p in pids],
                   SamplingParams(temperature=1.0, max_tokens=NTOK, logprobs=0, ignore_eos=True, seed=0))

worst = 0.0
total_exact = total = 0
for i, out in enumerate(gen):
    comp = out.outputs[0]
    gids = list(comp.token_ids)
    dec = [comp.logprobs[j][gids[j]].logprob for j in range(len(gids))]
    full = list(pids[i]) + gids
    # rescore this sequence ALONE (batch of 1)
    r = llm.generate([{"prompt_token_ids": full}],
                     SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))[0]
    pl = r.prompt_logprobs
    P = len(pids[i])
    pre = [pl[P + j][full[P + j]].logprob for j in range(len(gids))]
    diffs = [abs(dec[j] - pre[j]) for j in range(len(gids))]
    m = max(diffs)
    worst = max(worst, m)
    total_exact += sum(1 for d in diffs if d == 0.0)
    total += len(diffs)
    print(f"seq {i}: max|decode_batched - rescore_alone| = {m:.3e}  exact0={sum(1 for d in diffs if d==0.0)}/{len(diffs)}",
          flush=True)

print(f"\nBATCHED-vs-ALONE over {N} seqs: WORST max={worst:.3e}  total exact0={total_exact}/{total}", flush=True)
print("RESULT:", "ENGINE BATCH-INVARIANT (bitwise)" if worst == 0.0 else
      f"ENGINE BATCH-VARIANCE -> residual source (worst {worst:.3e})", flush=True)
