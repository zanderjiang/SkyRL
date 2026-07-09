"""Decode==prefill bitwise parity for the SkyRL zero-KL engine path (gptmodel_vllm + CUSTOM backend).

Unlike mimo_parity_test.py (which uses the standalone mimo_megatron_vllm wrapper), this drives the
ACTUAL SkyRL package code: skyrl.backends.skyrl_train.zerokl.gptmodel_vllm.GPTModelVLLMWrapper built
with the LOCAL layer spec (SKYRL_ZEROKL_LOCAL_SPEC=1) + the skyrl zerokl varlen_backend (num_splits=1).
If this is bitwise (max==0), the engine half of the SkyRL integration produces true zero-KL rollouts.

Run on the zero-KL nightly venv:
    SKYRL_ZERO_KL=1 SKYRL_ZEROKL_LOCAL_SPEC=1 VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    HF_HOME=/mnt/local_storage/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 \
    /mnt/local_storage/zerokl-nightly-venv/bin/python skyrl_engine_parity_test.py
"""
import os

os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

N = int(os.environ.get("PARITY_NTOK", "256"))
MODEL = os.environ.get("ZEROKL_MODEL", "/mnt/local_storage/models/MiMo-7B-RL")
# Qwen3.5 (hybrid GDN): temp 1.0 exercises the sampler, max_num_seqs sizes the open-chunk buffers,
# and limit_mm_per_prompt keeps vLLM's startup profiling out of the (absent) vision processor.
TEMP = float(os.environ.get("PARITY_TEMP", "0.0"))
MAX_NUM_SEQS = int(os.environ.get("PARITY_MAX_NUM_SEQS", "8"))
MM_ZERO = os.environ.get("PARITY_MM_ZERO") == "1"

import torch  # noqa: E402
import vllm.envs as vllm_envs  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

# Register the CUSTOM num_splits=1 varlen backend + the GPTModel wrapper, exactly as
# vllm_engine.setup_envvars_for_vllm does for a real SkyRL run.
import skyrl.backends.skyrl_train.zerokl.varlen_backend as varlen_backend  # noqa: E402,F401
from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import (  # noqa: E402
    register_gptmodel_to_vllm, VLLM_MODEL_NAME)
from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env  # noqa: E402

print(f"=== SkyRL zero-KL engine parity | torch {torch.__version__} | vllm {__import__('vllm').__version__} "
      f"| BI={vllm_envs.VLLM_BATCH_INVARIANT} N={N} ===", flush=True)
print("varlen backend usable:", varlen_backend.register_varlen_custom_backend(), flush=True)
apply_vllm_zerokl_env()
register_gptmodel_to_vllm()

llm = LLM(
    model=MODEL,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
    dtype="bfloat16",
    enforce_eager=True,
    gpu_memory_utilization=0.55,
    max_model_len=2048,
    max_num_seqs=MAX_NUM_SEQS,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    trust_remote_code=True,
    **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if MM_ZERO else {}),
)

tok = llm.get_tokenizer()
# (A) coherent generation
prompt = "The capital of France is"
pids = tok(prompt, add_special_tokens=False).input_ids
g = llm.generate([{"prompt_token_ids": pids}],
                 SamplingParams(temperature=0.0, max_tokens=40, logprobs=0))[0]
print("[GEN]", repr(prompt), "->", repr(tok.decode(g.outputs[0].token_ids)), flush=True)

# (B) bitwise decode==prefill over N tokens
prompt2 = ("Solve step by step: a train travels at 60 mph for 2.5 hours, then 40 mph for 1.5 "
           "hours. Explain how to find the total distance and discuss factors affecting it.")
p2 = tok(prompt2, add_special_tokens=False).input_ids[:64]
out = llm.generate([{"prompt_token_ids": p2}],
                   SamplingParams(temperature=TEMP, top_p=1.0, seed=0, max_tokens=N,
                                  logprobs=0, ignore_eos=True))[0]
comp = out.outputs[0]
gen_ids = list(comp.token_ids)
n = len(gen_ids)
decode_lps = [comp.logprobs[i][gen_ids[i]].logprob for i in range(n)]
full = list(p2) + gen_ids
out2 = llm.generate([{"prompt_token_ids": full}],
                    SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))[0]
pl = out2.prompt_logprobs
P = len(p2)
prefill_lps = [pl[P + i][full[P + i]].logprob for i in range(n)]
diffs = [abs(decode_lps[i] - prefill_lps[i]) for i in range(n)]
maxd = max(diffs)
exact0 = sum(1 for d in diffs if d == 0.0)
print(f"\nMAX |decode - prefill| over {n} tokens = {maxd:.6e}", flush=True)
print(f"tokens EXACT 0.0: {exact0}/{n}", flush=True)
print("RESULT:", "BITWISE-IDENTICAL (max==0)" if maxd == 0.0 else f"DRIFT max={maxd:.3e}", flush=True)
