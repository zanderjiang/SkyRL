"""Pinpoint WHICH stage of the live trainer forward introduces the zero-KL residual.

rescore_seq_probe.py proved: engine-prefill(saved tokens) == ROLLOUT bitwise, but the live TRAINER
action_log_probs differ (diffuse ~0.014). The bare-GPTModel forward (engine_trainer_parity_harness)
matches the engine bitwise on greedy tokens. This script scores the SAVED live tokens through a freshly
built trainer GPTModel (same spec as megatron_worker: local spec + varlen swap + batch-invariant) TWO
ways and compares each to the saved rollout (== engine, ground truth):

  (1) BARE forward + fp32 log_softmax+gather        (my harness method)
  (2) from_parallel_logits_to_logprobs on same logits (the LIVE extraction)

If (1) == rollout bitwise -> the GPTModel forward is correct; the residual is the LIVE machinery
(remove_left_padding / Float16Module / forward_backward_func / the extraction). If (2) != (1), the
extraction (from_parallel_logits_to_logprobs) is the culprit. If (1) != rollout too, the forward itself
diverges on these (temp-1.0/tail) tokens -> a numeric/precision issue in the GPTModel forward.

One free GPU:
    HF_HUB_OFFLINE=1 HF_HOME=/mnt/local_storage/hf CUDA_VISIBLE_DEVICES=<g> \
    SKYRL_ZERO_KL=1 SKYRL_ZEROKL_LOCAL_SPEC=1 VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VARLEN_FORCE_NUM_SPLITS_1=1 /mnt/local_storage/zerokl-nightly-venv/bin/python probe_trainer_forward.py
"""
import os

os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VARLEN_FORCE_NUM_SPLITS_1", "1")

PROBE = os.environ.get("ZK_SEQ_PROBE", "/mnt/local_storage/zerokl_seq_probe.pt")
MODEL = os.environ.get("ZEROKL_MODEL", "/mnt/local_storage/models/MiMo-7B-RL")

import torch  # noqa: E402
# vLLM batch-invariant must be enabled BEFORE the forward (engine build normally does it); do it here.
import skyrl.backends.skyrl_train.zerokl.varlen_backend as _vb  # noqa: E402,F401
_vb.register_varlen_custom_backend()

d = torch.load(PROBE, map_location="cpu")
seq_full = d["sequences"]
am = d["attention_mask"].bool()
na = int(d["num_actions"])
rol = d["rollout_action_logprobs"].float()[0]     # == engine (proven)
live_trn = d["action_log_probs"].float()[0]        # the live trainer (buggy)
b = 0
real = seq_full[b][am[b]].tolist()
Lr = len(real)
P = Lr - na
print(f"probe: real={Lr} prompt={P} na={na} | saved max|live_trn-rol|={float((live_trn-rol).abs().max()):.4f}",
      flush=True)

dev = torch.device("cuda:0")
torch.cuda.set_device(0)
from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.models.gpt_provider import local_layer_spec  # noqa: E402
from transformers import AutoConfig  # noqa: E402

b_ = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
mp = b_.to_megatron_provider(load_weights=True)
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

from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import (  # noqa: E402
    enable_trainer_batch_invariant, swap_trainer_core_attention_varlen)
enable_trainer_batch_invariant()
swap_trainer_core_attention_varlen(_bare)

from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import _zerokl_scoring_ctx  # noqa: E402

seq = torch.tensor(real, dtype=torch.long, device=dev).unsqueeze(0)
pos = torch.arange(Lr, device=dev).unsqueeze(0)
_bare.eval()
with torch.no_grad(), _zerokl_scoring_ctx():
    logits = _bare(input_ids=seq, position_ids=pos, attention_mask=None)[0]  # [Lr, V] (bf16 or fp32)
print(f"logits dtype={logits.dtype} shape={tuple(logits.shape)}", flush=True)

# (1) BARE fp32 log_softmax+gather
dlp = torch.log_softmax(logits.float(), dim=-1)
tgt = torch.tensor(real[P:P + na], device=dev)
bare_lp = dlp[P - 1:P + na - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()

# (2) from_parallel_logits_to_logprobs (the LIVE extraction) on the SAME logits
try:
    from megatron.post_training.algos.distillation import from_parallel_logits_to_logprobs  # type: ignore
    _has_fp = True
except Exception:
    try:
        from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import from_parallel_logits_to_logprobs  # noqa: E402
        _has_fp = True
    except Exception as _e:
        print(f"(2) from_parallel import failed: {_e}", flush=True)
        _has_fp = False


def stat(x, y):
    dd = (x - y).abs()
    return f"max={float(dd.max()):.4f} mean={float(dd.mean()):.5f} frac>0.05={float((dd>0.05).float().mean()):.4f} exact0={int((dd==0).sum())}/{x.numel()}"


print("\n=== compare to ROLLOUT (== engine, ground truth) ===", flush=True)
print(f"(1) BARE fp32 log_softmax      vs rollout : {stat(bare_lp, rol)}", flush=True)
print(f"    BARE fp32 log_softmax      vs live_trn: {stat(bare_lp, live_trn)}", flush=True)

if _has_fp:
    import megatron.core.parallel_state as mpu  # noqa: E402
    lg = logits.unsqueeze(0)  # [1, Lr, V]
    sq = seq
    try:
        fp = from_parallel_logits_to_logprobs(
            lg, sq, vocab_start_index=0, vocab_end_index=logits.shape[-1],
            tp_group=mpu.get_tensor_model_parallel_group(), inference_only=True, cp_group=None,
        )
        fp_resp = fp[0, -na:].float().cpu()
        print(f"(2) from_parallel_logits       vs rollout : {stat(fp_resp, rol)}", flush=True)
        print(f"    from_parallel vs BARE fp32 log_softmax : {stat(fp_resp, bare_lp)}", flush=True)
    except Exception as _e:
        print(f"(2) from_parallel call failed: {type(_e).__name__}: {_e}", flush=True)

print("\nVERDICT:", flush=True)
_bare_bit = float((bare_lp - rol).abs().max()) == 0.0
if _bare_bit:
    print("  BARE forward == rollout BITWISE -> GPTModel forward is CORRECT; residual is the LIVE MACHINERY "
          "(remove_left_padding / Float16Module / fbf / extraction). Compare (2) above to find the stage.", flush=True)
else:
    print(f"  BARE forward != rollout (max {float((bare_lp-rol).abs().max()):.4f}) -> the GPTModel forward itself "
          "diverges from the engine on these tokens (precision/kernel on temp-1.0/tail tokens), NOT just machinery.",
          flush=True)
