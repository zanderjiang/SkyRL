"""END-TO-END forward divergence localizer (engine-via-vLLM vs trainer-standalone).

Every isolated component is already proven bitwise (engine decode==prefill; engine vs trainer
weights byte-identical; vLLM fused logprob == manual log_softmax; paged vs non-paged varlen_attn).
So the ~5% / max~0.22 per-token logprob residual on real long sequences must live in the FULL
36-layer forward INTEGRATION. This harness reproduces (or fails to reproduce) it in-process:

  STEP 5: engine REAL forward (llm.generate prompt_logprobs, the rollout numerics) vs trainer
          STANDALONE megatron forward (bare gpt2 + varlen swap + batch-invariant) on the SAME
          ~1500-1900 token [prompt+response]. Report max / mean / frac>0.05 / worst tokens.
  LOCALIZE: if step 5 diverges, run wrapper.gpt AND gpt2 STANDALONE (both bare, varlen swap),
            hook every decoder layer, find FIRST divergent layer + op + magnitude. If the two
            STANDALONE runs are bitwise-identical, the cause is the vLLM execution path vs a bare
            module call (not any megatron op) -- report THAT conclusion.

Token alignment (rigorous):
  eng_lp[j]  = out.prompt_logprobs[j][seq[j]].logprob   = logP(seq[j] | seq[:j])   for j in 1..L-1
  gpt2 logits[1,L,V] -> from_parallel_logits_to_logprobs(logits, seq) rolls target -1 and returns
  [1, L-1] where out[0, i] = logP(seq[i+1] | seq[:i+1]).  So trn_lp[j] = out[0, j-1].
"""
import os
os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VARLEN_FORCE_NUM_SPLITS_1", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = os.environ.get("ZEROKL_MODEL", "/mnt/local_storage/models/MiMo-7B-RL")
GEN_TOKENS = int(os.environ.get("ZEROKL_GEN_TOKENS", "1500"))

import torch  # noqa: E402
import skyrl.backends.skyrl_train.zerokl.varlen_backend as varlen_backend  # noqa: E402,F401
from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import (  # noqa: E402
    register_gptmodel_to_vllm, VLLM_MODEL_NAME, find_inprocess_gptmodel)
from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

# ---------------------------------------------------------------------------
# (1) build the engine exactly as rope_weight_diff_harness.py does
# ---------------------------------------------------------------------------
apply_vllm_zerokl_env()
register_gptmodel_to_vllm()
print("=== building engine ===", flush=True)
llm = LLM(model=MODEL, hf_overrides={"architectures": [VLLM_MODEL_NAME]}, attention_backend="CUSTOM",
          dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.45, max_model_len=2048,
          enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True)
wrapper = find_inprocess_gptmodel(llm)
assert wrapper is not None and hasattr(wrapper, "gpt"), "could not reach in-process wrapper"
DEV = next(wrapper.gpt.parameters()).device
print(f"engine wrapper reached. device={DEV}", flush=True)

# ---------------------------------------------------------------------------
# (2) pick a fixed long token sequence: tokenize a real math prompt, greedily
#     generate ~GEN_TOKENS tokens with the engine -> realistic [prompt+response].
# ---------------------------------------------------------------------------
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
PROMPT = (
    "Let f(x) = x^3 - 6x^2 + 11x - 6. Find all real roots of f, then compute the area "
    "between the curve and the x-axis over [1, 3]. Show every step of your reasoning in "
    "full detail, including factoring, integration, and arithmetic."
)
try:
    prompt_ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                         add_generation_prompt=True, tokenize=True)
except Exception:
    prompt_ids = tok(PROMPT)["input_ids"]
prompt_ids = list(map(int, prompt_ids))
print(f"prompt len={len(prompt_ids)}; generating up to {GEN_TOKENS} tokens (greedy)...", flush=True)
gen = llm.generate([{"prompt_token_ids": prompt_ids}],
                   SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS))[0]
resp_ids = list(map(int, gen.outputs[0].token_ids))
seq = prompt_ids + resp_ids
L = len(seq)
assert L < 2048, f"sequence length {L} exceeds max_model_len 2048"
print(f"[SEQ] L={L} (prompt={len(prompt_ids)} response={len(resp_ids)})", flush=True)

# ---------------------------------------------------------------------------
# (3) ENGINE per-token logprobs = the REAL vLLM forward (same numerics as rollout)
# ---------------------------------------------------------------------------
out = llm.generate([{"prompt_token_ids": seq}],
                   SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))[0]
plp = out.prompt_logprobs
assert plp is not None and len(plp) == L, f"prompt_logprobs len {None if plp is None else len(plp)} != L {L}"
eng_lp = torch.full((L,), float("nan"))
for j in range(1, L):
    d = plp[j]
    assert seq[j] in d, f"token {seq[j]} not in prompt_logprobs[{j}]"
    eng_lp[j] = d[seq[j]].logprob
print(f"[ENGINE] collected {int((~eng_lp.isnan()).sum())} per-token logprobs via vLLM prompt_logprobs", flush=True)

# ---------------------------------------------------------------------------
# (4) build the SECOND trainer-style bridge GPTModel (gpt2), exactly like the base harness
# ---------------------------------------------------------------------------
print("\n=== building trainer-style bridge GPTModel (gpt2) ===", flush=True)
from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.models.gpt_provider import local_layer_spec  # noqa: E402
from transformers import AutoConfig  # noqa: E402
b = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
mp = b.to_megatron_provider(load_weights=True)
mp.tensor_model_parallel_size = 1
mp.pipeline_model_parallel_size = 1
mp.pipeline_dtype = torch.bfloat16
mp.apply_rope_fusion = False
mp.gradient_accumulation_fusion = False
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
gpt2 = mp.provide_distributed_model(wrap_with_ddp=False)
gpt2 = gpt2[0].module if hasattr(gpt2[0], "module") else gpt2[0]

# weight sanity (should be byte-identical -- already proven, cheap to reconfirm)
with torch.no_grad():
    eng_w = dict(wrapper.gpt.named_parameters())
    worst_w = max((float((eng_w[n].float() - p.float().to(eng_w[n].device)).abs().max())
                   for n, p in gpt2.named_parameters() if n in eng_w), default=0.0)
print(f"[WEIGHTS] worst |gpt2 - wrapper.gpt| = {worst_w:.3e} (expect 0)", flush=True)

# swap gpt2's attention to the exact engine kernel + enable batch-invariant non-attn ops
from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import (  # noqa: E402
    swap_trainer_core_attention_varlen, enable_trainer_batch_invariant)
enable_trainer_batch_invariant()
swap_trainer_core_attention_varlen(gpt2)

import megatron.core.parallel_state as mpu  # noqa: E402
from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (  # noqa: E402
    from_parallel_logits_to_logprobs)


def bare_forward_logits(model, seq_ids):
    """Run a bare standalone megatron GPTModel forward (the trainer's exact unpadded call)."""
    model.eval()
    ids = torch.tensor([seq_ids], device=DEV, dtype=torch.long)
    pos = torch.arange(len(seq_ids), device=DEV).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=ids, position_ids=pos, attention_mask=None)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    # normalize to [1, L, V]
    if logits.dim() == 3 and logits.shape[0] == len(seq_ids) and logits.shape[1] == 1:
        logits = logits.transpose(0, 1)  # [s,b,v] -> [b,s,v]
    return logits.contiguous()


# ---------------------------------------------------------------------------
# (4b) TRAINER per-token logprobs from a STANDALONE megatron forward of gpt2
# ---------------------------------------------------------------------------
print("\n=== trainer standalone forward (gpt2) ===", flush=True)
logits = bare_forward_logits(gpt2, seq)
print(f"[TRAINER] gpt2 logits shape={tuple(logits.shape)}", flush=True)
tp_grp = mpu.get_tensor_model_parallel_group()
tp_rank = mpu.get_tensor_model_parallel_rank()
seq_t = torch.tensor([seq], device=logits.device, dtype=torch.long)
trn_raw = from_parallel_logits_to_logprobs(
    logits, seq_t,
    vocab_start_index=tp_rank * logits.shape[-1],
    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
    tp_group=tp_grp, inference_only=True, cp_group=None, chunk_size=None,
)[0]  # [L-1], trn_raw[i] = logP(seq[i+1] | seq[:i+1])
trn_lp = torch.full((L,), float("nan"))
trn_lp[1:L] = trn_raw[0:L - 1].float().cpu()

# ---------------------------------------------------------------------------
# (5) DIFF
# ---------------------------------------------------------------------------
valid = (~eng_lp.isnan()) & (~trn_lp.isnan())
idx = valid.nonzero(as_tuple=True)[0]
d = (trn_lp[idx] - eng_lp[idx]).abs()
n = d.numel()
print("\n================= STEP 5: ENGINE-via-vLLM vs TRAINER-standalone =================", flush=True)
print(f"[DIFF] tokens compared = {n}  (L={L})", flush=True)
print(f"[DIFF] max  = {float(d.max()):.6f}", flush=True)
print(f"[DIFF] mean = {float(d.mean()):.6e}", flush=True)
for thr in (0.05, 0.1, 0.2):
    c = int((d > thr).sum())
    print(f"[DIFF] count > {thr:<4} = {c}  ({100.0 * c / n:.2f}%)", flush=True)
order = torch.argsort(d, descending=True)[:12]
print("[DIFF] worst tokens (seq_pos, token_id, trn, eng, |diff|):", flush=True)
for o in order.tolist():
    j = int(idx[o])
    print(f"    pos={j:5d} tok={seq[j]:7d} trn={float(trn_lp[j]):+.5f} "
          f"eng={float(eng_lp[j]):+.5f} |d|={float(d[o]):.5f}", flush=True)
REPRODUCES = float(d.max()) > 0.05

# ---------------------------------------------------------------------------
# LOCALIZE: only if step 5 shows divergence
# ---------------------------------------------------------------------------
print("\n================= LOCALIZE =================", flush=True)
if not REPRODUCES:
    print("[LOCALIZE] step 5 max <= 0.05 -> divergence did NOT reproduce in-process; nothing to localize.",
          flush=True)
else:
    # Run wrapper.gpt AND gpt2 BOTH standalone (bare, same varlen swap), hook every decoder layer,
    # find the FIRST layer where their hidden states diverge.
    print("[LOCALIZE] swapping wrapper.gpt attention -> varlen (to run it bare like gpt2)...", flush=True)
    swap_trainer_core_attention_varlen(wrapper.gpt)

    def unwrap(m):
        inner = m
        for _ in range(4):
            if hasattr(inner, "decoder"):
                break
            inner = getattr(inner, "module", inner)
        return inner

    def run_with_hooks(model, seq_ids):
        inner = unwrap(model)
        caps = {}
        handles = []

        def mk(i):
            def hook(mod, args, output):
                hs = output[0] if isinstance(output, (tuple, list)) else output
                caps[i] = hs.detach().float().cpu()
            return hook
        for i, layer in enumerate(inner.decoder.layers):
            handles.append(layer.register_forward_hook(mk(i)))
        try:
            lg = bare_forward_logits(model, seq_ids)
        finally:
            for h in handles:
                h.remove()
        return caps, lg

    print("[LOCALIZE] running gpt2 standalone with per-layer hooks...", flush=True)
    cap_trn, _ = run_with_hooks(gpt2, seq)
    print("[LOCALIZE] running wrapper.gpt standalone with per-layer hooks...", flush=True)
    cap_eng, wrap_logits = run_with_hooks(wrapper.gpt, seq)

    # Direct logprob proof: wrapper.gpt BARE vs the SAME wrapper.gpt driven by vLLM (eng_lp).
    wlp_raw = from_parallel_logits_to_logprobs(
        wrap_logits, seq_t,
        vocab_start_index=tp_rank * wrap_logits.shape[-1],
        vocab_end_index=(tp_rank + 1) * wrap_logits.shape[-1],
        tp_group=tp_grp, inference_only=True, cp_group=None, chunk_size=None,
    )[0]
    wrap_lp = torch.full((L,), float("nan"))
    wrap_lp[1:L] = wlp_raw[0:L - 1].float().cpu()
    dwe = (wrap_lp[idx] - eng_lp[idx]).abs()
    dwt = (wrap_lp[idx] - trn_lp[idx]).abs()
    print(f"[LOCALIZE] wrapper.gpt BARE vs ENGINE(vLLM): max={float(dwe.max()):.6f} mean={float(dwe.mean()):.3e} "
          f"frac>0.05={100.0 * float((dwe > 0.05).sum()) / n:.2f}%", flush=True)
    print(f"[LOCALIZE] wrapper.gpt BARE vs gpt2 BARE   : max={float(dwt.max()):.6e} mean={float(dwt.mean()):.3e} "
          f"(expect ~0 -- same code+weights)", flush=True)

    nlayers = min(len(cap_trn), len(cap_eng))
    print(f"[LOCALIZE] comparing {nlayers} decoder-layer hidden states (gpt2 vs wrapper.gpt, both bare):",
          flush=True)
    first_div = None
    for i in range(nlayers):
        a, bb = cap_trn[i], cap_eng[i]
        m = float((a - bb).abs().max())
        if i < 3 or m > 0:
            print(f"    layer {i:2d}: max|diff| = {m:.3e}  shape={tuple(a.shape)}", flush=True)
        if m > 0 and first_div is None:
            first_div = (i, m)
    if first_div is None:
        print("\n[LOCALIZE] CONCLUSION: gpt2-standalone and wrapper.gpt-standalone are BITWISE-IDENTICAL "
              "across all 36 layers. The divergence is NOT reproducible with a bare module call -> the "
              "cause is the vLLM EXECUTION PATH (how vLLM drives wrapper.gpt: paged-KV attention "
              "metadata / chunked prefill / cuda-graph / dtype) vs a bare full-sequence forward, NOT any "
              "megatron op or weight.", flush=True)
    else:
        print(f"\n[LOCALIZE] CONCLUSION: first divergent decoder layer = {first_div[0]} "
              f"(max|diff|={first_div[1]:.3e}). Divergence reproduces standalone at this layer.", flush=True)

print("\n=== DONE ===", flush=True)
