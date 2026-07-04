"""Decisive weights-vs-data-vs-forward-input test for the zero-KL residual.

The live run (SKYRL_ZEROKL_SEQ_PROBE=1) saves ONE residual-carrying micro-batch to
/mnt/local_storage/zerokl_seq_probe.pt: the EXACT token sequence the trainer scored, its trainer
logprobs (action_log_probs), and the engine rollout logprobs (rollout_action_logprobs) the residual is
measured against. This script rebuilds the zero-KL engine and RE-SCORES those very tokens via prefill,
then does a 3-way compare at every response position:

    trainer (saved)  vs  engine-prefill(saved tokens)  vs  rollout (saved)

Interpretation (forward is already proven bitwise for identical weights + identical tokens):
  * engine-prefill(saved) == trainer  but  != rollout  -> the ROLLOUT logprob was computed on a
    DIFFERENT context than the trainer scored (generator token-assembly / alignment). DATA bug.
  * engine-prefill(saved) == rollout  but  != trainer  -> the TRAINER forward-input differs
    (position_ids / padding / mask). FORWARD-INPUT bug.
  * all three equal (bitwise)                            -> no residual on these tokens (weights fine).
  * engine-prefill(saved) != BOTH                        -> weights differ at generation vs now.

Run on the nightly venv, ONE free GPU:
    HF_HUB_OFFLINE=1 HF_HOME=/mnt/local_storage/hf CUDA_VISIBLE_DEVICES=<g> \
    SKYRL_ZERO_KL=1 SKYRL_ZEROKL_LOCAL_SPEC=1 SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=1 \
    VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VARLEN_FORCE_NUM_SPLITS_1=1 \
    SKYRL_ZEROKL_NO_CHUNKED_PREFILL=1 \
    /mnt/local_storage/zerokl-nightly-venv/bin/python rescore_seq_probe.py
"""
import os

os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VARLEN_FORCE_NUM_SPLITS_1", "1")
os.environ.setdefault("SKYRL_ZEROKL_NO_CHUNKED_PREFILL", "1")

PROBE = os.environ.get("ZK_SEQ_PROBE", "/mnt/local_storage/zerokl_seq_probe.pt")
MODEL = os.environ.get("ZEROKL_MODEL", "/mnt/local_storage/models/MiMo-7B-RL")
GMU = float(os.environ.get("ZK_ENGINE_GPU_MEM_UTIL", "0.40"))

import torch  # noqa: E402

d = torch.load(PROBE, map_location="cpu")
seq = d["sequences"]            # [B, L] padded
am = d["attention_mask"].bool()  # [B, L]
na = int(d["num_actions"])
trn = d["action_log_probs"].float()          # [B, na]
rol = d["rollout_action_logprobs"].float()   # [B, na]
B, L = seq.shape
print(f"loaded probe: B={B} L={L} na={na} | saved max|trn-rol|={float((trn-rol).abs().max()):.4f}", flush=True)

# Reconstruct the FIRST sample's real (unpadded) token ids. Real tokens where attention_mask==1.
b = 0
real = seq[b][am[b]].tolist()
Lr = len(real)
# response = last na real tokens; prompt = the rest
P = Lr - na
assert P > 0, f"prompt len {P} <= 0"
print(f"real tokens={Lr} prompt={P} response={na}", flush=True)

import vllm.envs as vllm_envs  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
import skyrl.backends.skyrl_train.zerokl.varlen_backend as varlen_backend  # noqa: E402,F401
from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import register_gptmodel_to_vllm, VLLM_MODEL_NAME  # noqa: E402
from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env  # noqa: E402

print(f"=== engine build | BI={vllm_envs.VLLM_BATCH_INVARIANT} ===", flush=True)
varlen_backend.register_varlen_custom_backend()
apply_vllm_zerokl_env()
register_gptmodel_to_vllm()
# ZK_BLOCK_SIZE: paged KV-cache block size. TorchTitan's bitwise-parity test uses 256 ("align with
# FA2"); SkyRL uses vLLM's default (16). Larger blocks ~= closer to contiguous KV -> may shrink the
# paged-vs-nonpaged drift vs the non-paged trainer. 0 => vLLM default.
_bs = int(os.environ.get("ZK_BLOCK_SIZE", "0"))
_llm_kw = dict(model=MODEL, hf_overrides={"architectures": [VLLM_MODEL_NAME]}, attention_backend="CUSTOM",
               dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=GMU, max_model_len=Lr + 8,
               max_num_seqs=2, enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True)
if _bs > 0:
    _llm_kw["block_size"] = _bs
print(f"=== engine block_size={_bs or 'default'} ===", flush=True)
llm = LLM(**_llm_kw)

# Prefill-rescore the exact real token sequence: prompt_logprobs over the full seq.
out = llm.generate([{"prompt_token_ids": real}],
                   SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))[0]
pl = out.prompt_logprobs
eng = [pl[P + i][real[P + i]].logprob for i in range(na)]  # engine-prefill logprob per response tok
eng_t = torch.tensor(eng)

d_et = (eng_t - trn[b]).abs()   # engine-prefill(saved) vs trainer(saved)
d_er = (eng_t - rol[b]).abs()   # engine-prefill(saved) vs rollout(saved)
d_tr = (trn[b] - rol[b]).abs()  # trainer(saved) vs rollout(saved) == the run's residual


def stat(x):
    return f"max={float(x.max()):.4f} mean={float(x.mean()):.5f} frac>0.05={float((x>0.05).float().mean()):.4f} exact0={int((x==0).sum())}/{x.numel()}"


print("\n--- 3-way compare on the EXACT saved tokens ---", flush=True)
print(f"[residual]  trainer   vs rollout   : {stat(d_tr)}", flush=True)
print(f"[A] engine-prefill(saved) vs trainer : {stat(d_et)}", flush=True)
print(f"[B] engine-prefill(saved) vs rollout : {stat(d_er)}", flush=True)

et_bit = float(d_et.max()) == 0.0
er_bit = float(d_er.max()) == 0.0
print("\nVERDICT:", flush=True)
if float(d_tr.max()) == 0.0:
    print("  no residual on these tokens (weights + inputs fine here).", flush=True)
elif et_bit and not er_bit:
    print("  engine-prefill(saved) == TRAINER but != ROLLOUT  ->  the ROLLOUT logprobs were computed on a "
          "DIFFERENT context than the trainer scored: GENERATOR TOKEN-ASSEMBLY / ALIGNMENT bug (data).", flush=True)
elif er_bit and not et_bit:
    print("  engine-prefill(saved) == ROLLOUT but != TRAINER  ->  the TRAINER forward-input differs "
          "(position_ids / padding / attention_mask): FORWARD-INPUT bug.", flush=True)
elif not et_bit and not er_bit:
    print("  engine-prefill(saved) != BOTH  ->  weights at generation differ from weights now, OR the saved "
          "tokens don't reconstruct either trajectory. Inspect worst positions below.", flush=True)
else:
    print("  all three ~bitwise -> residual not reproduced on these tokens.", flush=True)

# show worst residual positions with all three streams
top = torch.topk(d_tr, k=min(10, na)).indices.tolist()
print("\nworst residual positions (i=resp idx):", flush=True)
for i in top:
    print(f"  i={i}/{na} tok={real[P+i]} trn={float(trn[b][i]):.4f} eng_prefill={eng[i]:.4f} "
          f"rol={float(rol[b][i]):.4f} | d(eng,trn)={float(d_et[i]):.4f} d(eng,rol)={float(d_er[i]):.4f}",
          flush=True)
