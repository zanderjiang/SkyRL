"""Layer-by-layer trace: dump EVERY intermediate tensor of the engine forward AND the trainer forward on
the SAME real sequence (identical weights), then diff in forward order to pinpoint exactly which op
introduces the ~0.014 rollout-vs-train divergence. No speculation -- capture both, compare.

Both the engine (gptmodel_vllm) and the trainer are the SAME Megatron GPTModel (local spec, WrappedTorchNorm,
same MLP/GEMMs); the engine only swaps core_attention -> vLLM Attention + _PositionIndexedRoPE, the trainer
swaps -> TorchVarlenCoreAttn + stock rotary. So per-submodule intermediate tensors ([L,1,H] sbhd on both)
are directly comparable by module name. Weights are native-synced engine->trainer so the ONLY difference is
the forward path. Uses the saved real rollout tokens (zerokl_seq_probe.pt) which reproduce the live 0.014.

Engine on cuda:0, trainer on cuda:1. enforce_eager (no cudagraphs) so hooks fire.
    CUDA_VISIBLE_DEVICES=6,7 SKYRL_ZERO_KL=1 SKYRL_ZEROKL_LOCAL_SPEC=1 SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=1 \
    VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VARLEN_FORCE_NUM_SPLITS_1=1 \
    SKYRL_ZEROKL_NO_CHUNKED_PREFILL=1 HF_HUB_OFFLINE=1 HF_HOME=/mnt/local_storage/hf \
    /mnt/local_storage/zerokl-nightly-venv/bin/python trace_layerwise.py
"""
import os

os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VARLEN_FORCE_NUM_SPLITS_1", "1")
os.environ.setdefault("SKYRL_ZEROKL_NO_CHUNKED_PREFILL", "1")

MODEL = os.environ.get("ZEROKL_MODEL", "/mnt/local_storage/models/MiMo-7B-RL")
PROBE = os.environ.get("ZK_SEQ_PROBE", "/mnt/local_storage/zerokl_seq_probe.pt")
GMU = float(os.environ.get("ZK_ENGINE_GPU_MEM_UTIL", "0.35"))

import torch  # noqa: E402

NGEN = int(os.environ.get("ZK_NGEN", "1000"))  # temp-1.0 gen length (reproduces the real-rollout divergence)
# real/L/P/na/rol are set AFTER the engine is built, from a temp-1.0 generation (see below).
real = None; L = None; P = None; na = None; rol = None


# ---- hook machinery: capture the FIRST prefill-shaped output (a dim == L) per module ----
def make_store():
    return {}


def register(gpt, store, order):
    idx = [0]
    handles = []
    for name, mod in gpt.named_modules():
        if name == "":
            continue

        def hook(m, inp, out, _name=name):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t):
                return
            if _name in store:
                return
            if L in tuple(t.shape):
                store[_name] = t.detach().float().cpu()
                if _name not in order:
                    order[_name] = idx[0]; idx[0] += 1
        handles.append(mod.register_forward_hook(hook))

        if name.endswith("core_attention"):
            def prehook(m, args, _name=name):
                for j, a in enumerate(args):
                    k = f"{_name}::IN{j}"
                    if torch.is_tensor(a) and k not in store:
                        store[k] = a.detach().float().cpu()
            handles.append(mod.register_forward_pre_hook(prehook))
    return handles


order = {}

# ================= ENGINE (cuda:0) =================
import vllm.envs as vllm_envs  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
import skyrl.backends.skyrl_train.zerokl.varlen_backend as varlen_backend  # noqa: E402,F401
from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import (  # noqa: E402
    register_gptmodel_to_vllm, VLLM_MODEL_NAME, find_inprocess_gptmodel)
from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env  # noqa: E402

print(f"=== build engine | BI={vllm_envs.VLLM_BATCH_INVARIANT} ===", flush=True)
varlen_backend.register_varlen_custom_backend()
apply_vllm_zerokl_env()
register_gptmodel_to_vllm()
llm = LLM(model=MODEL, hf_overrides={"architectures": [VLLM_MODEL_NAME]}, attention_backend="CUSTOM",
          dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=GMU, max_model_len=NGEN + 256,
          max_num_seqs=2, enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True)
wrapper = find_inprocess_gptmodel(llm)
tok = llm.get_tokenizer()
# --- ZK_USE_PROBE=1: load the EXACT live tokens + rollout logprobs from the seq probe (the sequence
#     that shows 0.014 in the live run) and run them through this bitwise single-process setup. If it
#     reproduces 0.014 -> token-dependent (layer dump localizes); if bitwise -> distributed-only. ---
if os.environ.get("ZK_USE_PROBE") == "1":
    _d = torch.load(os.environ.get("ZK_SEQ_PROBE", "/mnt/local_storage/zerokl_seq_probe.pt"), map_location="cpu")
    _am = _d["attention_mask"].bool()
    na = int(_d["num_actions"])
    rol = _d["rollout_action_logprobs"].float()[0]
    real = _d["sequences"][0][_am[0]].tolist()
    L = len(real); P = L - na
    _sv = _d.get("action_log_probs")
    _svmax = float((_sv.float()[0] - rol).abs().max()) if _sv is not None else -1.0
    print(f"USING LIVE PROBE TOKENS: L={L} prompt={P} na={na} saved max|trn-rol|={_svmax:.4f}", flush=True)
    _SKIP_GEN = True
else:
    _SKIP_GEN = False
# --- reproduce the LIVE generation: real AIME chat-templated prompts, BATCHED (continuous batching) ---
NBATCH = int(os.environ.get("ZK_NBATCH", "8"))
PICK = int(os.environ.get("ZK_PICK", "0"))
# Varied prompts of DIFFERENT lengths so they finish at different steps -> real continuous batching
# (batch composition changes each decode step), exactly like the live DP8 run.
_texts = [
    "Solve step by step and box the answer: find the minimum of 1/a+1/b+1/c for a+b+c=1.",
    "Explain in detail, step by step, why the sum of the first n odd numbers is n squared, with proof.",
    "A train travels 60 mph for 2.5 hours then 40 mph for 1.5 hours; compute total distance and discuss.",
    "Prove the AM-GM inequality for three variables and give a worked numerical example, step by step.",
    "Define a_1=3, a_{n+1}=2a_n+1; find a closed form, prove by induction, and compute a_10 in detail.",
    "Compute the number of ways to tile a 2xN board with dominoes; derive the recurrence and solve it.",
    "Find all real solutions to x^4-5x^2+4=0 by factoring, and explain each algebraic step carefully.",
    "Explain the Euclidean algorithm for gcd, prove it terminates, and trace gcd(1071,462) step by step.",
]
if not _SKIP_GEN:
    _prompt_ids = [tok(_texts[i % len(_texts)], add_special_tokens=False).input_ids for i in range(NBATCH)]
    _gs = llm.generate([{"prompt_token_ids": p} for p in _prompt_ids],
                       SamplingParams(temperature=1.0, top_p=1.0, max_tokens=NGEN, logprobs=0, seed=0, ignore_eos=True))
    _g = _gs[PICK]
    _pids = _prompt_ids[PICK]
    _gen = list(_g.outputs[0].token_ids)
    rol = torch.tensor([_g.outputs[0].logprobs[i][_gen[i]].logprob for i in range(len(_gen))])  # engine DECODE lp
    real = list(_pids) + _gen
    L = len(real); na = len(_gen); P = len(_pids)
    print(f"generated (batch of {NBATCH}, pick {PICK}): L={L} prompt={P} na={na}", flush=True)

eng_store = make_store()
eng_handles = register(wrapper.gpt, eng_store, order)
# engine PREFILL of the full real sequence (== decode rollout, proven) -> hooks capture every intermediate
out = llm.generate([{"prompt_token_ids": real}],
                   SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))[0]
for h in eng_handles:
    h.remove()
pl = out.prompt_logprobs
eng_lp = torch.tensor([pl[P + i][real[P + i]].logprob for i in range(na)])
print(f"engine captured {len(eng_store)} module tensors; engine-prefill vs decode-rollout "
      f"max={float((eng_lp - rol).abs().max()):.4f} (should be ~0)", flush=True)

# ================= TRAINER (cuda:1) =================
trainer_dev = torch.device("cuda:1")
torch.cuda.set_device(1)
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
_gpt = mp.provide_distributed_model(wrap_with_ddp=False)
_bare = _gpt[0].module if hasattr(_gpt[0], "module") else _gpt[0]
for _ in range(4):
    if hasattr(_bare, "decoder"):
        break
    _bare = getattr(_bare, "module", _bare)

# native-sync engine weights -> trainer so ONLY the forward path differs. ZK_NO_SYNC=1 SKIPS the sync
# so the trainer keeps its OWN independent bridge HF load (like the live run + probe_trainer_forward) --
# if that diverges from the engine, the two independent weight LOADS differ (root cause = weights).
from skyrl.backends.skyrl_train.zerokl.native_weight_sync import extract_native_weights, load_native_weights  # noqa
if os.environ.get("ZK_NO_SYNC") == "1":
    print("[TRACE] ZK_NO_SYNC=1 -> trainer keeps its OWN bridge weights (no sync from engine)", flush=True)
    # verify element-wise whether the two independent loads even match
    _e = dict(wrapper.gpt.named_parameters()); _t = dict(_bare.named_parameters())
    _wmax = 0.0; _wn = ""; _nd = 0
    for _nm, _p in _t.items():
        if _nm in _e:
            _dw = (_e[_nm].float().to(_p.device) - _p.float()).abs().max().item()
            if _dw > _wmax:
                _wmax = _dw; _wn = _nm
            if _dw > 0:
                _nd += 1
    print(f"[TRACE] independent bridge loads: params_with_diff={_nd}/{len(_t)} worst={_wmax:.3e} @ {_wn}", flush=True)
else:
    load_native_weights(_bare, iter(list(extract_native_weights(wrapper.gpt, dtype=torch.bfloat16))), strict=False)
from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import (  # noqa: E402
    enable_trainer_batch_invariant, swap_trainer_core_attention_varlen)
enable_trainer_batch_invariant()
swap_trainer_core_attention_varlen(_bare)

from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import _zerokl_scoring_ctx  # noqa
trn_store = make_store()
trn_handles = register(_bare, trn_store, order)
seq = torch.tensor(real, dtype=torch.long, device=trainer_dev).unsqueeze(0)
pos = torch.arange(L, device=trainer_dev).unsqueeze(0)
_bare.eval()
# ZK_GRAD=1: run the trainer forward with grad ENABLED (as the live forward_backward_func(forward_only=True)
# does per scoring_mode docstring) instead of no_grad -- grad-mode can pick different cuBLAS GEMM algos.
import contextlib as _cl  # noqa: E402
_gctx = _cl.nullcontext() if os.environ.get("ZK_GRAD") == "1" else torch.no_grad()
if os.environ.get("ZK_GRAD") == "1":
    print("[TRACE] trainer forward with GRAD ENABLED (matches fbf forward_only)", flush=True)
with _gctx, _zerokl_scoring_ctx():
    logits = _bare(input_ids=seq, position_ids=pos, attention_mask=None)[0].float()
for h in trn_handles:
    h.remove()
dlp = torch.log_softmax(logits, dim=-1)
tgt = torch.tensor(real[P:P + na], device=trainer_dev)
trn_lp = dlp[P - 1:P + na - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
print(f"trainer captured {len(trn_store)} module tensors; trainer vs rollout "
      f"max={float((trn_lp - rol).abs().max()):.4f} mean={float((trn_lp - rol).abs().mean()):.5f}", flush=True)

# save the CLEAN bare-forward intermediates so compare_layertrace.py can diff them against the LIVE
# fbf-forward intermediates (trace_live_trainer.pt) to localize the distributed-machinery divergence.
if os.environ.get("ZK_SAVE_CLEAN") == "1":
    torch.save({"store": trn_store, "order": order, "P": P, "na": na, "L": L},
               "/mnt/local_storage/trace_clean_trainer.pt")
    print("[TRACE] saved clean intermediates -> trace_clean_trainer.pt", flush=True)


# ================= DIFF in forward order =================
def to_LH(t):
    # bring the L dim to front, flatten the rest -> [L, F]
    shp = tuple(t.shape)
    ax = shp.index(L)
    t = t.movedim(ax, 0)
    return t.reshape(L, -1)


common = [n for n in sorted(order, key=lambda x: order[x]) if n in eng_store and n in trn_store]
print(f"\n=== {len(common)} common modules; per-module diff over RESPONSE region [{P}:{L}] ===", flush=True)
print(f"{'idx':>4} {'module':<52} {'shape':<16} {'max':>10} {'mean':>10}", flush=True)
rows = []
for n in common:
    try:
        te = to_LH(eng_store[n])[P:]
        tt = to_LH(trn_store[n])[P:]
        if te.shape != tt.shape:
            continue
        dd = (te - tt).abs()
        rows.append((order[n], n, tuple(eng_store[n].shape), float(dd.max()), float(dd.mean())))
    except Exception:
        continue
# print decoder-layer residual outputs + the first few layers' submodules in detail
for idx, n, shp, mx, mn in rows:
    mark = "  <== FIRST NONZERO" if mn > 0 and all(r[4] == 0 for r in rows if r[0] < idx) else ""
    print(f"{idx:>4} {n:<52} {str(shp):<16} {mx:>10.3e} {mn:>10.3e}{mark}", flush=True)

# summarize: which module type first introduces nonzero, and the growth across decoder layers
layer_out = [(o, n, mx, mn) for o, n, shp, mx, mn in rows if n.startswith("decoder.layers.") and n.count(".") == 2]
print("\n=== residual-stream divergence after each decoder layer ===", flush=True)
for o, n, mx, mn in sorted(layer_out):
    print(f"  {n:<28} max={mx:.3e} mean={mn:.3e}", flush=True)
