"""Combined ENGINE+TRAINER zero-KL parity harness (the decisive divergence localizer).

The standalone loop (dapo_zerokl_nightly) shows engine decode==engine prefill BITWISE, and the
full SkyRL run's FWDPROBE shows the trainer's OWN forward machinery is bitwise (~1e-6). Yet the real
run's DIFF (engine rollout logprob vs TRAINER scoring logprob) is max ~0.2-1.0 on ~5% of long-seq
tokens -- i.e. NOT zero KL. Prior harnesses only proved engine==trainer forward on a SHORT (49-tok)
single sequence. This harness reproduces the comparison the way the Ray actors build the two models,
on a LONG real rollout, with IDENTICAL (native-synced) weights, to localize the residual:

  (A) engine DECODE logprobs vs engine PREFILL-rescore   -> is the engine self-consistent at length?
  (B) native-sync engine.gpt -> trainer GPTModel; weights bitwise identical?
  (C) engine PREFILL logprobs vs TRAINER-forward logprobs -> THE residual, per-token, localized:
      where do the outliers sit (high positions? sharp/low-entropy tokens?), and how big.

Both models are built EXACTLY as SkyRL builds them at TP=1:
  - engine: skyrl.backends.skyrl_train.zerokl.gptmodel_vllm (local spec + CUSTOM varlen num_splits=1)
  - trainer: a local-spec bridge GPTModel (== megatron_worker.init_configs at TP1) + the worker's
    swap_trainer_core_attention_varlen + enable_trainer_batch_invariant; scored with the bare-GPTModel
    recipe FWDPROBE uses (attention_mask=None, plain fp32 log_softmax+gather), which == the worker's
    forward_backward_func path to ~1e-6.

Two GPUs (engine on cuda:0, trainer GPTModel on cuda:1) so the two 14GB models don't co-OOM a GPU
that another job already half-fills. Launch (CUDA_VISIBLE_DEVICES picks the free pair):

    SKYRL_ZERO_KL=1 SKYRL_ZEROKL_LOCAL_SPEC=1 SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=1 \
    VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VARLEN_FORCE_NUM_SPLITS_1=1 \
    SKYRL_ZEROKL_NO_CHUNKED_PREFILL=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6,7 \
    PARITY_NTOK=1500 \
    /mnt/local_storage/zerokl-nightly-venv/bin/python engine_trainer_parity_harness.py
"""
import os

os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VARLEN_FORCE_NUM_SPLITS_1", "1")
os.environ.setdefault("SKYRL_ZEROKL_NO_CHUNKED_PREFILL", "1")

N = int(os.environ.get("PARITY_NTOK", "1500"))          # generated response length
MODEL = os.environ.get("ZEROKL_MODEL", "/mnt/local_storage/models/MiMo-7B-RL")
ENGINE_DEV = int(os.environ.get("ZK_ENGINE_DEV", "0"))   # index into CUDA_VISIBLE_DEVICES
TRAINER_DEV = int(os.environ.get("ZK_TRAINER_DEV", "1"))
GPU_MEM_UTIL = float(os.environ.get("ZK_ENGINE_GPU_MEM_UTIL", "0.32"))
# Force a LONG prompt so response tokens sit at HIGH absolute positions (~2000-8000), matching the live
# run (2048-tok prompt) -- the harness previously only reached abs-pos ~1583 (short prompt) where it was
# bitwise. ZK_PROMPT_LEN>0 tiles a filler passage to that many prompt tokens.
PROMPT_LEN = int(os.environ.get("ZK_PROMPT_LEN", "0"))
MAX_LEN = int(os.environ.get("ZK_MAX_MODEL_LEN", str(PROMPT_LEN + N + 96)))

import torch  # noqa: E402
import vllm.envs as vllm_envs  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

# Register the CUSTOM num_splits=1 varlen backend + the GPTModel wrapper, exactly as
# vllm_engine.setup_envvars_for_vllm does for a real SkyRL run.
import skyrl.backends.skyrl_train.zerokl.varlen_backend as varlen_backend  # noqa: E402,F401
from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import (  # noqa: E402
    register_gptmodel_to_vllm, VLLM_MODEL_NAME, find_inprocess_gptmodel)
from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env  # noqa: E402


def banner(s):
    print(f"\n========== {s} ==========", flush=True)


banner(f"BUILD ENGINE | torch {torch.__version__} | vllm {__import__('vllm').__version__} "
       f"| BI={vllm_envs.VLLM_BATCH_INVARIANT} N={N} MAX_LEN={MAX_LEN}")
print("varlen backend usable:", varlen_backend.register_varlen_custom_backend(), flush=True)
apply_vllm_zerokl_env()
register_gptmodel_to_vllm()

llm = LLM(
    model=MODEL,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
    dtype="bfloat16",
    enforce_eager=True,
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=MAX_LEN,
    max_num_seqs=2,           # harness scores ONE sequence; avoids the 1024-req sampler-warmup OOM
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    trust_remote_code=True,
)
tok = llm.get_tokenizer()
wrapper = find_inprocess_gptmodel(llm)
assert wrapper is not None and hasattr(wrapper, "gpt"), "could not reach in-process GPTModel wrapper"
eng_dev = next(wrapper.gpt.parameters()).device
print(f"engine wrapper reached; engine.gpt on {eng_dev}", flush=True)

# ---------------------------------------------------------------------------------------------
# (A) Generate ONE long real rollout; engine DECODE logprobs vs engine PREFILL-rescore.
# ---------------------------------------------------------------------------------------------
banner("(A) ENGINE decode vs prefill at length")
prompt = ("Let $a,b,c$ be positive reals with $a+b+c=1$. Solve this step by step, showing all "
          "algebra, then give the final boxed answer: find the minimum of "
          "$\\frac{1}{a}+\\frac{1}{b}+\\frac{1}{c}$ and prove it with the AM-HM inequality, then "
          "generalize to $n$ variables and discuss when equality holds.")
pids = tok(prompt, add_special_tokens=False).input_ids
if PROMPT_LEN > 0:
    # tile a filler passage in FRONT so the real question ends the prompt; pad to PROMPT_LEN tokens.
    filler = tok("In mathematics, careful reasoning proceeds one deduction at a time. " * 40,
                 add_special_tokens=False).input_ids
    pids = (filler * ((PROMPT_LEN // max(1, len(filler))) + 1))[:PROMPT_LEN - len(pids)] + pids
P = len(pids)
gen = llm.generate([{"prompt_token_ids": pids}],
                   SamplingParams(temperature=0.0, max_tokens=N, logprobs=0, ignore_eos=True))[0]
comp = gen.outputs[0]
gen_ids = list(comp.token_ids)
n = len(gen_ids)
decode_lps = [comp.logprobs[i][gen_ids[i]].logprob for i in range(n)]
print(f"prompt_len={P} generated n={n}; sample decode: {[round(x,3) for x in decode_lps[:6]]}", flush=True)

full = list(pids) + gen_ids
L = len(full)
out2 = llm.generate([{"prompt_token_ids": full}],
                    SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))[0]
pl = out2.prompt_logprobs
prefill_lps = [pl[P + i][full[P + i]].logprob for i in range(n)]
dp = [abs(decode_lps[i] - prefill_lps[i]) for i in range(n)]
print(f"[A] engine decode vs prefill: max={max(dp):.3e} mean={sum(dp)/n:.3e} "
      f"exact0={sum(1 for d in dp if d == 0.0)}/{n}", flush=True)

# ---------------------------------------------------------------------------------------------
# (B) Build the TRAINER GPTModel the way megatron_worker does (TP1 local spec), on cuda:TRAINER_DEV,
#     then NATIVE-SYNC the engine's weights into it (real native_weight_sync.py) -> identical weights.
# ---------------------------------------------------------------------------------------------
banner("(B) BUILD TRAINER GPTModel + native sync")
trainer_dev = torch.device(f"cuda:{TRAINER_DEV}")
torch.cuda.set_device(TRAINER_DEV)
from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.models.gpt_provider import local_layer_spec  # noqa: E402
from transformers import AutoConfig  # noqa: E402

b = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
mp = b.to_megatron_provider(load_weights=True)
mp.tensor_model_parallel_size = 1
mp.pipeline_model_parallel_size = 1
mp.expert_model_parallel_size = 1
mp.expert_tensor_parallel_size = 1
mp.pipeline_dtype = torch.bfloat16
mp.apply_rope_fusion = False
mp.gradient_accumulation_fusion = False
# Match megatron_worker.init_configs (provider.attention_backend = "flash"). The engine build already
# configured the NVTE_* env for flash; leaving this at the default "auto" trips megatron-core's
# _set_attention_backend assert (NVTE_FUSED_ATTN==0 vs expected 1). The actual kernel is replaced by
# swap_trainer_core_attention_varlen below, so this only needs to pass the __init__ env check.
mp.attention_backend = "flash"
mp.transformer_layer_spec = local_layer_spec
if getattr(mp, "mtp_num_layers", None):
    mp.mtp_num_layers = None
_hf = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
_rp = getattr(_hf, "rope_parameters", None) or getattr(_hf, "rope_scaling", None)
if isinstance(_rp, dict) and _rp.get("rope_theta"):
    mp.rotary_base = _rp["rope_theta"]
elif getattr(_hf, "rope_theta", None):
    mp.rotary_base = _hf.rope_theta
mp.finalize()
_gpt_list = mp.provide_distributed_model(wrap_with_ddp=False)
trainer_top = _gpt_list[0]
trainer_gpt = trainer_top.module if hasattr(trainer_top, "module") else trainer_top
# unwrap to the bare GPTModel (the one with .decoder), as FWDPROBE / native_weight_sync do.
_bare = trainer_gpt
for _ in range(4):
    if hasattr(_bare, "decoder"):
        break
    _bare = getattr(_bare, "module", _bare)
print(f"trainer GPTModel built on {next(_bare.parameters()).device} "
      f"(rotary_base={getattr(mp, 'rotary_base', '?')})", flush=True)

# --- native sync: engine.gpt -> trainer_gpt (the REAL sync code), then verify bitwise identical ---
from skyrl.backends.skyrl_train.zerokl.native_weight_sync import (  # noqa: E402
    extract_native_weights, load_native_weights)
weights = list(extract_native_weights(wrapper.gpt, dtype=torch.bfloat16))
loaded = load_native_weights(_bare, iter(weights), strict=False)
print(f"native sync: {len(loaded)} tensors copied into trainer", flush=True)
# verify identity after sync
eng = dict(wrapper.gpt.named_parameters())
trn = dict(_bare.named_parameters())
worst = 0.0
worst_name = ""
ndiff = 0
for nme, p in trn.items():
    if nme not in eng:
        continue
    with torch.no_grad():
        d = (eng[nme].float().to(p.device) - p.float()).abs().max().item()
    if d > worst:
        worst, worst_name = d, nme
    if d > 0:
        ndiff += 1
print(f"[B] post-sync weights: params_with_diff={ndiff}/{len(trn)} worst={worst:.3e} @ {worst_name}",
      flush=True)

# --- apply the worker's trainer-side kernel patches (EXACTLY megatron_worker init_model) ---
from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import (  # noqa: E402
    enable_trainer_batch_invariant, swap_trainer_core_attention_varlen)
if os.environ.get("SKYRL_ZEROKL_BATCH_INVARIANT", "1") == "1":
    enable_trainer_batch_invariant()
# ZK_SKIP_VARLEN_SWAP=1 leaves the trainer on the local-spec DEFAULT attention (SDPA) instead of the
# engine's varlen num_splits=1 kernel -- to test whether SDPA-vs-varlen is the live 0.2-1.0 residual
# (i.e. whether the live trainer's varlen swap silently isn't applying).
if os.environ.get("ZK_SKIP_VARLEN_SWAP") == "1":
    print("[HARNESS] SKIPPING trainer varlen swap -> trainer uses local-spec DEFAULT (SDPA)", flush=True)
else:
    swap_trainer_core_attention_varlen(_bare)

# ---------------------------------------------------------------------------------------------
# (C) TRAINER forward on the SAME full sequence (bare GPTModel, attention_mask=None, fp32
#     log_softmax+gather == the FWDPROBE recipe). Compare to engine prefill + engine decode.
# ---------------------------------------------------------------------------------------------
banner("(C) TRAINER forward vs ENGINE (the zero-KL residual)")
from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import _zerokl_scoring_ctx  # noqa: E402

# --- RoPE localizer: is the engine's position-indexed RoPE == the trainer's stock fresh RoPE at HIGH
#     absolute positions? (rope_weight_diff_harness only checked pos 0-255.) If these diverge at high
#     positions, RoPE is the residual; if bitwise, the high-pos residual is ATTENTION accumulation. ---
try:
    _rope = wrapper._rope
    with torch.no_grad():
        _idx = torch.arange(L, device=_rope._emb_full.device)
        _indexed = _rope._emb_full[_idx].float()      # engine: precomputed cache indexed by abs pos
        _fresh = _rope._orig(L).float().to(_indexed.device)  # trainer stock: fresh orig(L)
        _rd = (_indexed - _fresh).abs()
        _perpos = _rd.flatten(1).max(dim=1).values     # [L] max rope diff per position
        _hi = int((_perpos[1600:] > 0).sum()) if L > 1600 else 0
        print(f"[C-RoPE] max|engine_indexed - stock_fresh| over {L} pos = {float(_rd.max()):.3e} "
              f"| #pos>1600 that differ = {_hi} | worst pos = {int(_perpos.argmax())}", flush=True)
except Exception as _e:
    print(f"[C-RoPE] check failed: {type(_e).__name__}: {_e}", flush=True)

seq = torch.tensor(full, dtype=torch.long, device=trainer_dev).unsqueeze(0)  # [1, L]
pos = torch.arange(L, device=trainer_dev).unsqueeze(0)                       # [1, L]
_bare.eval()
with torch.no_grad(), _zerokl_scoring_ctx():
    logits = _bare(input_ids=seq, position_ids=pos, attention_mask=None)[0].float()  # [L, V]
dlp = torch.log_softmax(logits, dim=-1)
# trainer logprob of response token at position P+i is dlp[P+i-1][token=full[P+i]]
tgt = torch.tensor(full[P:P + n], device=trainer_dev)
trainer_lps = dlp[P - 1:P + n - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).tolist()

dec = torch.tensor(decode_lps)
pre = torch.tensor(prefill_lps)
trn_t = torch.tensor(trainer_lps)
d_pre = (trn_t - pre).abs()    # trainer vs engine prefill (clean cross-kernel)
d_dec = (trn_t - dec).abs()    # trainer vs engine decode  (== the real-run DIFF metric)

print(f"[C] trainer vs engine PREFILL: max={float(d_pre.max()):.4f} mean={float(d_pre.mean()):.5f} "
      f"frac>0.05={float((d_pre > 0.05).float().mean()):.4f} exact0={int((d_pre == 0).sum())}/{n}",
      flush=True)
print(f"[C] trainer vs engine DECODE : max={float(d_dec.max()):.4f} mean={float(d_dec.mean()):.5f} "
      f"frac>0.05={float((d_dec > 0.05).float().mean()):.4f} exact0={int((d_dec == 0).sum())}/{n}",
      flush=True)

# localize the worst outliers vs engine prefill: position in response, sharpness, hi-position cluster
topk = torch.topk(d_pre, k=min(12, n)).indices.tolist()
print("\n[C] top trainer-vs-prefill outliers (i=resp_idx, abs_pos=P+i):", flush=True)
for i in topk:
    print(f"   i={i}/{n} abs_pos={P + i}/{L} d={float(d_pre[i]):.4f} "
          f"trn={trainer_lps[i]:.4f} pre_eng={prefill_lps[i]:.4f} dec_eng={decode_lps[i]:.4f} "
          f"(engine logprob {'SHARP' if prefill_lps[i] > -0.3 else 'soft'})", flush=True)
# where do the >0.05 outliers cluster along the sequence?
bad = (d_pre > 0.05).nonzero(as_tuple=True)[0]
if len(bad) > 0:
    fracs = (bad.float() / n)
    print(f"\n[C] {len(bad)} outliers(>0.05): resp-pos quartile spread min={float(fracs.min()):.2f} "
          f"median={float(fracs.median()):.2f} max={float(fracs.max()):.2f} "
          f"(1.0 => clustered at end / high positions)", flush=True)
    # are outliers on sharp tokens? correlate |d| with engine logprob
    sharp = (pre[bad] > -0.3).float().mean()
    print(f"[C] fraction of outliers on SHARP (logprob>-0.3) tokens: {float(sharp):.2f}", flush=True)

print("\nRESULT:", flush=True)
print(f"  (A) engine decode==prefill : {'BITWISE' if max(dp) == 0 else f'DRIFT {max(dp):.2e}'}", flush=True)
print(f"  (B) synced weights         : {'IDENTICAL' if worst == 0 else f'DIFFER {worst:.2e}'}", flush=True)
print(f"  (C) trainer==engine prefill: {'BITWISE ZERO-KL' if float(d_pre.max()) == 0 else f'RESIDUAL max={float(d_pre.max()):.3f}'}", flush=True)
