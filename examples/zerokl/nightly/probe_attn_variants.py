"""Test trainer-attention variants on the REAL 2000+ token rollout, vs the engine (rollout) ground truth.

Root cause: the trainer's non-paged `varlen_attn` (bf16) != the engine's paged `varlen_attn_out` (bf16)
on real rollouts (~0.014 diffuse, worse on tail tokens). This builds ONE trainer GPTModel on the saved
real probe sequence (2239 tok) and re-runs the full forward with the decoder-layer core_attention swapped
to each variant, comparing the resulting response logprobs to the saved rollout (== engine, proven
bitwise). Whichever variant drives max|trainer - rollout| -> 0 is the fix.

Variants:
  varlen_bf16  : current TorchVarlenCoreAttn (torch varlen_attn, contiguous, bf16)         [baseline]
  sdpa_bf16    : F.scaled_dot_product_attention, is_causal, bf16
  sdpa_fp32    : F.scaled_dot_product_attention in FP32 (q/k/v upcast) -> bf16 out          [precision test]

One free GPU. Same env as probe_trainer_forward.py.
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
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

# [EXP] Reproduce the trainer runtime's flash-attn dispatch. In the LIVE trainer no vLLM engine is built
# in-process, so torch's flash-attn impl is never activated (current_flash_attention_impl() is None) and
# varlen_attn falls back to a non-FA3 path -> the ~0.014 residual. In-process harnesses that built a vLLM
# engine silently activated FA3, masking the bug. ZK_ACTIVATE_FA3=1 activates FA3 here == the fix.
if os.environ.get("ZK_ACTIVATE_FA3", "0") == "1":
    from torch.nn.attention import activate_flash_attention_impl, current_flash_attention_impl  # noqa: E402
    if current_flash_attention_impl() != "FA3":
        activate_flash_attention_impl("FA3")
    print(f"[EXP] flash impl activated: {current_flash_attention_impl()}", flush=True)
else:
    from torch.nn.attention import current_flash_attention_impl  # noqa: E402
    print(f"[EXP] flash impl (unmodified): {current_flash_attention_impl()}", flush=True)
import skyrl.backends.skyrl_train.zerokl.varlen_backend as _vb  # noqa: E402,F401
_vb.register_varlen_custom_backend()

d = torch.load(PROBE, map_location="cpu")
am = d["attention_mask"].bool()
na = int(d["num_actions"])
rol = d["rollout_action_logprobs"].float()[0]
real = d["sequences"][0][am[0]].tolist()
Lr = len(real)
P = Lr - na
print(f"probe: real={Lr} prompt={P} na={na}", flush=True)

dev = torch.device("cuda:0")
torch.cuda.set_device(0)
from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.models.gpt_provider import local_layer_spec  # noqa: E402
from transformers import AutoConfig  # noqa: E402

_b = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
mp = _b.to_megatron_provider(load_weights=True)
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
_bare.eval()

from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import enable_trainer_batch_invariant  # noqa: E402
enable_trainer_batch_invariant()

cfg = _bare.config
NH = cfg.num_attention_heads
NKV = cfg.num_query_groups
HD = getattr(cfg, "kv_channels", cfg.hidden_size // cfg.num_attention_heads)
SCALE = HD ** -0.5
GQA = NH > NKV
import torch.nn.attention.varlen as _V  # noqa: E402


class VariantAttn(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        sq, b = query.shape[0], query.shape[1]
        if self.mode == "varlen_bf16":
            q = query.reshape(sq, NH, HD); k = key.reshape(sq, NKV, HD); v = value.reshape(sq, NKV, HD)
            cu = torch.tensor([0, sq], device=q.device, dtype=torch.int32)
            out = _V.varlen_attn(q, k, v, cu, cu, sq, sq, scale=SCALE, num_splits=1,
                                 enable_gqa=GQA, window_size=(-1, 0))
            if isinstance(out, tuple):
                out = out[0]
            return out.reshape(sq, b, NH * HD)
        if self.mode == "varlen_out_bf16":
            # the ENGINE's exact function (varlen_attn_out), non-paged. If this matches rollout, the
            # gap was varlen_attn vs varlen_attn_out and the fix is a one-line trainer change.
            q = query.reshape(sq, NH, HD); k = key.reshape(sq, NKV, HD); v = value.reshape(sq, NKV, HD)
            cu = torch.tensor([0, sq], device=q.device, dtype=torch.int32)
            o = torch.empty_like(q)
            res = _V.varlen_attn_out(o, q, k, v, cu, cu, sq, sq, scale=SCALE, num_splits=1,
                                     enable_gqa=GQA, window_size=(-1, 0))
            if isinstance(res, tuple):
                res = res[0]
            return res.reshape(sq, b, NH * HD)
        # SDPA variants: [1, nh, sq, hd]
        q = query.reshape(sq, NH, HD).permute(1, 0, 2).unsqueeze(0)
        k = key.reshape(sq, NKV, HD).permute(1, 0, 2).unsqueeze(0)
        v = value.reshape(sq, NKV, HD).permute(1, 0, 2).unsqueeze(0)
        if self.mode == "sdpa_fp32":
            qd = q.float(); kd = k.float(); vd = v.float()
        else:
            qd = q; kd = k; vd = v
        out = F.scaled_dot_product_attention(qd, kd, vd, is_causal=True, scale=SCALE, enable_gqa=GQA)
        out = out.squeeze(0).permute(1, 0, 2).reshape(sq, b, NH * HD).to(query.dtype)
        return out


def set_attn(mode):
    n = 0
    for layer in _bare.decoder.layers:
        sa = getattr(layer, "self_attention", None)
        if sa is not None:
            sa.core_attention = VariantAttn(mode)
            n += 1
    return n


from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import _zerokl_scoring_ctx  # noqa: E402

seq = torch.tensor(real, dtype=torch.long, device=dev).unsqueeze(0)
pos = torch.arange(Lr, device=dev).unsqueeze(0)
tgt = torch.tensor(real[P:P + na], device=dev)


def score(mode):
    set_attn(mode)
    with torch.no_grad(), _zerokl_scoring_ctx():
        logits = _bare(input_ids=seq, position_ids=pos, attention_mask=None)[0].float()
    dlp = torch.log_softmax(logits, dim=-1)
    lp = dlp[P - 1:P + na - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
    dd = (lp - rol).abs()
    print(f"[{mode:12s}] vs rollout: max={float(dd.max()):.4f} mean={float(dd.mean()):.5f} "
          f"frac>0.05={float((dd>0.05).float().mean()):.4f} exact0={int((dd==0).sum())}/{na}", flush=True)
    return float(dd.max())


print("\n=== trainer-attention variants vs ROLLOUT (engine ground truth) on the REAL 2239-tok seq ===", flush=True)
for m in ("varlen_bf16", "varlen_out_bf16", "sdpa_bf16", "sdpa_fp32"):
    score(m)
