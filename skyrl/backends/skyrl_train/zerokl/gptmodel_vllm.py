"""Run Megatron's GPTModel INSIDE vLLM (unified-model zero-KL route).

Mirrors TorchTitan's rl/models/vllm_wrapper.py + vllm_registry.py, but the "model" is a
Megatron-core GPTModel built via megatron.bridge instead of a TorchTitan model. vLLM becomes
pure runtime (scheduler, paged KV cache, sampling); the compute is GPTModel — the SAME model
the trainer runs, so per-op inputs are bitwise-identical and (with batch-invariant + the proven
prefill==decode parity) generation matches the trainer pass.

Pieces:
  - MegatronCoreAttnToVLLM: drop-in replacement for SelfAttention.core_attention that routes
    Megatron-layout (q,k,v) into vLLM's paged Attention layer.
  - GPTModelVLLMWrapper: implements vLLM's model interface (forward/compute_logits/
    get_input_embeddings/load_weights) over a bridge-built GPTModel with attention swapped.
  - register_gptmodel_to_vllm: ModelRegistry.register_model + a config parser so vLLM builds
    THIS class for our custom architecture name.

Scope: TP=1, enforce_eager (breakable cudagraph absent in this vLLM). Generator attention uses
vLLM's vendored flash backend (proven == TE-flash; torch 2.11 lacks torch-native paged varlen).
Weights arrive via native sync (no HF) -> load_weights is effectively a copy/no-op.
"""

from __future__ import annotations

import itertools
import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)

VLLM_MODEL_NAME = "MegatronGPTModelForCausalLM"
TORCHTITAN_LIKE_CONFIG_FORMAT = "megatron_gptmodel"


# --------------------------------------------------------------------------------------
# SkyRL-ZeroKL: register the engine-side bitwise kernels IN THE WORKER PROCESS at import time.
# With distributed_executor_backend="mp", vLLM spawns a worker subprocess that does NOT inherit
# the driver-process @register_backend(CUSTOM) / logprob patch applied in vllm_engine.setup_envvars
# (that runs in the engine ACTOR, not the spawned worker). The worker lazily imports THIS module to
# build the model (string ModelRegistry registration), so registering here -- BEFORE vLLM resolves
# the attention backend and before sampling -- guarantees the worker actually uses CUSTOM varlen
# (num_splits=1 => bitwise decode==prefill at all lengths) and the aten-logprob path. Without it the
# worker silently falls back to vLLM's default flash backend, whose split-K heuristic makes
# long-context decode != prefill -> the ~0.27 minibatch_rollout_logprobs_abs_diff outliers on the
# minority of long sequences (mean tiny, max ~0.27, min 0). Idempotent; no-op off the local-spec path.
if os.environ.get("SKYRL_ZEROKL_LOCAL_SPEC") == "1":
    try:
        from skyrl.backends.skyrl_train.zerokl import varlen_backend as _zk_varlen
        from skyrl.backends.skyrl_train.zerokl.vllm_patches import (
            patch_vllm_logprobs_batch_invariant as _zk_patch_lp,
        )

        _zk_custom_ok = _zk_varlen.register_varlen_custom_backend()
        _zk_patch_lp()
        try:
            with open("/mnt/local_storage/zerokl_probe.log", "a") as _zk_pf:
                _zk_pf.write(f"[ZEROKL-WORKER-INIT] pid={os.getpid()} imported gptmodel_vllm; "
                             f"CUSTOM varlen available={_zk_custom_ok}; logprob patch applied\n")
        except Exception:
            pass
        logger.info("[zerokl] worker-side init: CUSTOM varlen available=%s, logprob patch applied",
                    _zk_custom_ok)
    except Exception as _zk_e:  # pragma: no cover
        logger.warning("[zerokl] worker-side CUSTOM/logprob registration failed: %s", _zk_e)


# --------------------------------------------------------------------------------------
# core_attention replacement: Megatron interface -> vLLM paged Attention
# --------------------------------------------------------------------------------------
class MegatronCoreAttnToVLLM(nn.Module):
    """Replacement for ``SelfAttention.core_attention`` that uses vLLM's paged Attention.

    Megatron calls ``core_attention(query, key, value, attention_mask, attn_mask_type=...,
    attention_bias=..., packed_seq_params=...)`` with q/k/v in ``[sq, b, np, hn]`` (sbhd) and
    expects ``[sq, b, np*hn]`` back. Under vLLM, b==1 and vLLM owns the KV cache + causal mask,
    so we drop the batch dim, hand ``[tokens, heads, hn]`` to ``vllm.Attention``, and reshape
    back. q/k-norm and RoPE were already applied upstream by SelfAttention.
    """

    _layer_counter = itertools.count()

    def __init__(self, *, num_heads: int, num_kv_heads: int, head_dim: int, scale: float):
        super().__init__()
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.attention import Attention

        vllm_config = get_current_vllm_config()
        cache_config = getattr(vllm_config, "cache_config", None)
        layer_id = next(MegatronCoreAttnToVLLM._layer_counter)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.vllm_attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"decoder.layers.{layer_id}.self_attention.core_attention",
        )

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        # Megatron sbhd: [sq, b, np, hn]. vLLM's Attention wants FLATTENED 2D
        # [num_tokens, np*hn] (matches native Qwen3: self.attn(q,k,v) with q [T, q_size]).
        sq, b = query.shape[0], query.shape[1]
        q = query.reshape(sq * b, self.num_heads * self.head_dim).contiguous()
        k = key.reshape(sq * b, self.num_kv_heads * self.head_dim).contiguous()
        v = value.reshape(sq * b, self.num_kv_heads * self.head_dim).contiguous()
        out = self.vllm_attn(q, k, v)                  # [num_tokens, np*hn]
        return out.reshape(sq, b, self.num_heads * self.head_dim)


def swap_core_attention(gpt_modules, *, num_heads, num_kv_heads, head_dim, scale):
    """Replace every decoder layer's ``self_attention.core_attention`` with the vLLM adapter."""
    modules = gpt_modules if isinstance(gpt_modules, (list, tuple)) else [gpt_modules]
    n = 0
    for m in modules:
        inner = m.module if hasattr(m, "module") else m
        for layer in inner.decoder.layers:
            sa = getattr(layer, "self_attention", None)
            if sa is None:
                continue
            sa.core_attention = MegatronCoreAttnToVLLM(
                num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, scale=scale
            )
            n += 1
    logger.info("[zerokl] swapped core_attention -> vLLM paged Attention on %d layers", n)
    return n


class _PositionIndexedRoPE(nn.Module):
    """Wraps Megatron's RotaryEmbedding so the returned RoPE is indexed by vLLM's ABSOLUTE
    positions instead of sequence-index 0..L-1. Required for paged decode (1-token inputs whose
    true position is N). For prefill (positions==0..L-1) it reproduces the original exactly."""

    def __init__(self, orig, max_pos):
        super().__init__()
        self._orig = orig
        with torch.no_grad():
            self._emb_full = orig(max_pos)        # [max_pos, 1, 1, dim]
        self._positions = None

    def set_positions(self, positions):
        self._positions = positions

    def forward(self, max_seq_len, *args, **kwargs):
        if self._positions is not None:
            return self._emb_full.to(self._positions.device)[self._positions]  # [T,1,1,dim]
        return self._orig(max_seq_len, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)       # _orig (submodule), _positions, etc.
        except AttributeError:
            return getattr(self._orig, name)        # delegate get_rotary_seq_len/get_cos_sin/...


# --------------------------------------------------------------------------------------
# vLLM model wrapper over a bridge-built GPTModel
# --------------------------------------------------------------------------------------
class GPTModelVLLMWrapper(nn.Module):
    """vLLM model whose compute is a Megatron GPTModel (attention swapped to vLLM paged).

    Built per `register_gptmodel_to_vllm` closure (captures model_path). Implements the vLLM
    model interface. TP=1 bring-up; weights loaded via native sync (load_weights is a no-op
    that reports param names for vLLM's safety check, like TorchTitan's wrapper).
    """

    def __init__(self, *, vllm_config, prefix="", model_path=None, load_weights=None):
        super().__init__()
        from megatron.bridge import AutoBridge
        from megatron.core.transformer.enums import AttnBackend
        from skyrl.backends.skyrl_train.zerokl import (
            apply_megatron_zerokl_patches,
            make_zerokl_local_layer_spec,
            patch_olmoe_bridge_for_sequential_mlp,
            prepare_zerokl_moe,
        )

        # vLLM string-registration instantiates us with only (vllm_config, prefix); derive the
        # model path from the engine config in that case.
        if model_path is None:
            model_path = vllm_config.model_config.model
        # load_weights: bridge loads real HF weights at init (standalone). Under SkyRL the trainer
        # overwrites them via native sync, but loading at init is harmless (env override available).
        if load_weights is None:
            load_weights = os.environ.get("SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS", "1") == "1"
        # Same pre-bridge mapping retarget the trainer does (megatron_worker.init_configs): under the
        # SequentialMLP pin, OLMoE's grouped-GEMM-only expert mappings would match nothing.
        if os.environ.get("SKYRL_ZEROKL_LOCAL_SPEC") == "1":
            patch_olmoe_bridge_for_sequential_mlp()
        # Same `fla` facade the trainer installs: the engine builds the SAME GPTModel, so its GDN
        # layers must run the same ops. Idempotent -- normally already done by the zerokl package
        # __init__, which runs before this module's body.
        if os.environ.get("SKYRL_ZEROKL_GDN") == "1":
            from skyrl.backends.skyrl_train.zerokl import install_fla_shim

            install_fla_shim()
        b = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
        mp = b.to_megatron_provider(load_weights=load_weights)
        mp.tensor_model_parallel_size = 1
        mp.pipeline_model_parallel_size = 1
        mp.expert_model_parallel_size = 1
        mp.expert_tensor_parallel_size = 1
        mp.pipeline_dtype = torch.bfloat16
        mp.apply_rope_fusion = False
        mp.attention_backend = AttnBackend.flash
        mp.gradient_accumulation_fusion = False
        # Nightly no-TE stack: force Megatron LOCAL layer spec so the engine GPTModel uses the
        # SAME plain-torch ops as the local-spec trainer (megatron_worker SKYRL_ZEROKL_LOCAL_SPEC).
        # Bitwise zero-KL comes from VLLM_BATCH_INVARIANT + num_splits=1 attention over this
        # batch-invariant local-spec forward -- NOT from the TE-targeted megatron patches below.
        self._local_spec = os.environ.get("SKYRL_ZEROKL_LOCAL_SPEC") == "1"
        if self._local_spec:
            # MoE providers resolve to get_gpt_layer_local_spec(num_experts=..., grouped_gemm=False)
            # -> MoELayer(TopKRouter, SequentialMLP); dense providers to bridge's local_layer_spec.
            mp.transformer_layer_spec = make_zerokl_local_layer_spec(mp)
            print("[ZEROKL-WRAP] forced Megatron LOCAL layer spec (no TransformerEngine)", flush=True)
        # Disable MTP (multi-token prediction) to MATCH the trainer, which drops it for training
        # (megatron_worker.init_configs: mtp_num_layers -> None). MiMo-7B ships MTP layers; if the
        # engine keeps them but the trainer doesn't, the two GPTModels differ (and the local-spec MTP
        # block spec build itself fails). Both sides must be the plain backbone for bitwise zero-KL.
        if getattr(mp, "mtp_num_layers", None):
            print(f"[ZEROKL-WRAP] disabling MTP (mtp_num_layers={mp.mtp_num_layers} -> None) to match trainer",
                  flush=True)
            mp.mtp_num_layers = None
        # CRITICAL for zero-KL: mirror the trainer's transformers-v5 RoPE-base workaround
        # (megatron_worker make_megatron_provider). transformers v5 moves rope_theta into
        # rope_parameters, so megatron-bridge's CONFIG_MAPPING reads the now-missing config.rope_theta
        # and silently falls back to rotary_base=10000. The trainer fixes it; the engine MUST too,
        # else the engine runs RoPE at a different base than the trainer -> logprobs diverge.
        _hf = vllm_config.model_config.hf_config
        _rp = getattr(_hf, "rope_parameters", None) or getattr(_hf, "rope_scaling", None)
        if isinstance(_rp, dict) and _rp.get("rope_theta"):
            mp.rotary_base = _rp["rope_theta"]
        elif getattr(_hf, "rope_theta", None):
            mp.rotary_base = _hf.rope_theta
        print(f"[ZEROKL-WRAP] rotary_base set to {getattr(mp, 'rotary_base', '?')} (hf rope_theta="
              f"{getattr(_hf, 'rope_theta', None)}, rope_parameters={_rp})", flush=True)
        # MoE: pin the identical recipe the trainer pins (SequentialMLP + allgather + fp32 router +
        # no fusions) and install the deterministic combine / sorted router top-k. Both sides must
        # apply this or the two GPTModels are different models. No-op on dense providers.
        if self._local_spec:
            prepare_zerokl_moe(mp, side="ENGINE")
        mp.finalize()
        gpt = mp.provide_distributed_model(wrap_with_ddp=False)
        self._gpt_list = gpt
        self.gpt = gpt[0].module if hasattr(gpt[0], "module") else gpt[0]

        # numeric recipe. On the LOCAL-spec nightly stack the model is already plain torch
        # (SDPA/RMSNorm/F.linear) and batch-invariance comes entirely from VLLM_BATCH_INVARIANT=1
        # -- the apply_megatron_zerokl_patches helpers target TE kernels (fused norm / TE GEMM) that
        # do not exist here, so we skip them (this is exactly what the proven standalone build does).
        # On the production TE stack (flag unset) we keep the original patch path.
        if not self._local_spec:
            # skip aten re-registration: vLLM (VLLM_BATCH_INVARIANT=1) already registered mm/addmm/
            # _log_softmax/mean; we only add the TE GEMM/RMSNorm + RoPE patches here.
            apply_megatron_zerokl_patches(skip_aten_registration=True)

        # swap attention -> vLLM paged
        cfg = self.gpt.config
        head_dim = getattr(cfg, "kv_channels", cfg.hidden_size // cfg.num_attention_heads)
        swap_core_attention(
            self.gpt,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_query_groups,
            head_dim=head_dim,
            scale=head_dim ** -0.5,
        )

        # RoPE-by-absolute-position: GPTModel computes RoPE for sequence-index 0..L-1, but vLLM
        # paged decode feeds 1-token inputs whose true position is N. Index a precomputed RoPE
        # cache by vLLM's `positions` so decode rotates at the right angle (else decode != prefill).
        max_pos = int(getattr(vllm_config.model_config, "max_model_len", 8192))
        self._rope = _PositionIndexedRoPE(self.gpt.rotary_pos_emb, max_pos)
        self.gpt.rotary_pos_emb = self._rope
        # Patch vLLM's sampler logprob kernel to the trainer's exact formula HERE (in the mp worker
        # process where the sampler runs) -- doing it in the engine actor (setup_envvars_for_vllm) does
        # NOT reach the mp worker. This is the last piece for bitwise rollout==train (the fused Triton
        # logprob kernel otherwise bypasses aten and diverges from the trainer on a few tokens).
        if self._local_spec:
            try:
                from skyrl.backends.skyrl_train.zerokl.vllm_patches import patch_vllm_logprobs_batch_invariant
                patch_vllm_logprobs_batch_invariant()
            except Exception as _e:  # pragma: no cover
                print(f"[ZEROKL-WRAP] logprob patch failed: {type(_e).__name__}: {_e}", flush=True)
        self._fwd_probe_done = False
        with torch.no_grad():
            _w = next((p for n, p in self.gpt.named_parameters() if "weight" in n), None)
            _wn = float(_w.float().norm()) if _w is not None else -1.0
        print(f"[ZEROKL-WRAP] built: model_path={model_path} load_weights={load_weights} "
              f"gpt_params={sum(1 for _ in self.gpt.named_parameters())} first_w_norm={_wn:.3f}", flush=True)

    def embed_input_ids(self, input_ids):
        # vLLM VllmModel protocol requires this exact name.
        return self.gpt.embedding(input_ids=input_ids.unsqueeze(0), position_ids=None)

    def get_input_embeddings(self, input_ids):
        return self.embed_input_ids(input_ids)

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        # vLLM varlen [total_tokens] -> Megatron [b=1, seq]. GPTModel applies RoPE internally;
        # attention is the swapped vLLM paged layer (ignores attention_mask, uses vLLM metadata).
        tokens = input_ids.unsqueeze(0)
        pos = positions.unsqueeze(0)
        # SkyRL-ZeroKL: checksum the weights the ENGINE forward actually consumes, on the first
        # forward after each native sync (flag set by WorkerWrap.load_weights). Same bf16 basis as
        # the RECEIVER post-sync total — unequal means the forward reads different storage than
        # the module params the sync updated.
        if getattr(self, "_zk_fwd_ck_pending", False):
            self._zk_fwd_ck_pending = False
            try:
                from skyrl.backends.skyrl_train.zerokl.native_weight_sync import param_abs_sum_bf16

                print(f"[ZEROKL-ENGFWD] first forward after sync: gpt abs-sum={param_abs_sum_bf16(self.gpt):.6f}", flush=True)
            except Exception as _e:
                print(f"[ZEROKL-ENGFWD] checksum failed: {_e}", flush=True)
        # Per-tensor weight dump (first forward) to compare ENGINE runtime weights (post native-sync)
        # to the TRAINER's forward weights offline -> find which params the sync diverges on.
        if os.environ.get("SKYRL_ZEROKL_BISECT") == "1" and not getattr(self, "_zk_wdump_done", False):
            self._zk_wdump_done = True
            try:
                from skyrl.backends.skyrl_train.zerokl.weight_fingerprint import fingerprint_line
                with open("/mnt/local_storage/zerokl_eng_w.txt", "w") as _wf:
                    for _nm, _p in self.gpt.named_parameters():
                        if _p.device.type == "meta":
                            continue
                        # Element-wise fingerprint (abs-sum + sum-of-squares + ramp-dot): an abs-sum
                        # match alone can mask per-element sync drift; all three matching is conclusive.
                        _wf.write(fingerprint_line(_nm, _p))
                print("[ZEROKL-WDUMP] engine weights (3-fp) -> zerokl_eng_w.txt", flush=True)
            except Exception as _e:
                print(f"[ZEROKL-WDUMP] engine failed: {_e}", flush=True)
        self._rope.set_positions(positions.reshape(-1))  # absolute positions for RoPE
        out = self.gpt(input_ids=tokens, position_ids=pos, attention_mask=None)
        # GPTModel(post_process) returns logits [b, s, vocab]; we expose them via forward and
        # let compute_logits pass through (pragmatic bring-up; TODO split hidden vs lm_head).
        if out.dim() == 3:
            out = out.reshape(-1, out.shape[-1])
        self._fwd_count = getattr(self, "_fwd_count", 0) + 1
        if os.environ.get("SKYRL_ZEROKL_BISECT") == "1" and self._fwd_count in (1, 2, 50):
            # Engine RUNTIME weight checksum over the NON-MTP params (the 255 the trainer syncs),
            # in the SAME formula the sender uses (bf16 -> float -> double -> abs -> sum). Compare to
            # [ZEROKL-CKSUM] SENDER ... checksum=X: if equal, the engine generates with exactly the
            # trainer's synced weights; if different, the sync/cumem delivers stale/wrong weights.
            with torch.no_grad():
                _s, _n = 0.0, 0
                for _nm, _p in self.gpt.named_parameters():
                    if _nm.startswith("mtp."):
                        continue
                    if _p.device.type == "meta":
                        continue
                    _s += float(_p.float().double().abs().sum()); _n += 1
            print(f"[ZEROKL-BISECT] ENGINE runtime non-MTP cksum={_s:.6f} (n={_n}) "
                  f"fwd#{self._fwd_count}  [compare to SENDER checksum]", flush=True)
            try:
                with open("/mnt/local_storage/zerokl_probe.log", "a") as _pf:
                    _pf.write(f"BISECT-ENGINE cksum={_s:.6f} n={_n} fwd#{self._fwd_count}\n")
            except Exception:
                pass
        if self._fwd_count in (1, 30, 100, 300):
            with torch.no_grad():
                _wn = float(next((p for n, p in self.gpt.named_parameters() if "weight" in n)).float().norm())
                lastlogits = out[-1].float()
                lp = torch.log_softmax(lastlogits, dim=-1)
                ent = float(-(lp.exp() * lp).sum())  # entropy of the ENGINE's own output dist
                top = lastlogits.topk(3)
            print(f"[ZEROKL-WRAP] forward#{self._fwd_count}: out={tuple(out.shape)} first_w_norm={_wn:.3f} "
                  f"out_entropy={ent:.3f} top3_ids={top.indices.tolist()}", flush=True)
        return out

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return hidden_states  # forward already produced fp32 logits via Float16Module

    def load_weights(self, weights_iter):
        # SkyRL-ZeroKL native sync: copy any NATIVE-named incoming params straight into self.gpt
        # (this is what propagates the trainer's updated weights to the rollout each step). The
        # incoming names are NATIVE at trainer-sync time; at vLLM BUILD time vLLM calls this with
        # HF-checkpoint names (which miss) -- harmless, since the bridge already populated self.gpt.
        # We ALWAYS return the full param-name set so vLLM's "all weights initialized" check passes
        # (the model is fully initialized by the bridge regardless of what this call matched).
        all_names = {"gpt." + n for n, _ in self.gpt.named_parameters()}
        dst = dict(self.gpt.named_parameters())
        dst.update(dict(self.gpt.named_buffers()))
        loaded, missed = 0, []
        with torch.no_grad():
            for name, tensor in weights_iter:
                dest = dst.get(name)
                if dest is None and name.startswith("module."):
                    dest = dst.get(name[len("module."):])  # tolerate residual wrapper prefix
                if dest is None:
                    if len(missed) < 3:
                        missed.append(name)
                    continue
                d = dest.full_tensor() if hasattr(dest, "full_tensor") else dest
                if tuple(d.shape) != tuple(tensor.shape):
                    continue
                dest.copy_(tensor.to(dest.dtype))
                loaded += 1
        with torch.no_grad():
            _wn = float(next((p for n, p in self.gpt.named_parameters() if "weight" in n)).float().norm())
        print(f"[ZEROKL-WRAP] load_weights: copied {loaded} native tensors into gpt "
              f"(non-native skipped, e.g. {missed}); first_w_norm={_wn:.3f}", flush=True)
        return all_names


_WRAPPER_IMPORT_PATH = "skyrl.backends.skyrl_train.zerokl.gptmodel_vllm:GPTModelVLLMWrapper"


def register_gptmodel_to_vllm(model_path: str | None = None):
    """Register the GPTModel-backed wrapper with vLLM. Call before engine init.

    Uses vLLM's STRING registration form (``module:ClassName``) so the registration survives
    across vLLM's mp/async worker subprocesses (each worker lazily imports the class). The
    wrapper derives the model path from ``vllm_config`` at build time, so no closure is needed.
    Set the engine's ``hf_overrides={"architectures": [VLLM_MODEL_NAME]}`` so vLLM builds this
    class instead of the model's native architecture.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    if model_path is not None:  # legacy closure form (single-process / standalone tests)
        class _Wrapper(GPTModelVLLMWrapper):
            def __init__(self, *, vllm_config, prefix=""):
                super().__init__(model_path=model_path, vllm_config=vllm_config, prefix=prefix)
        _Wrapper.__name__ = VLLM_MODEL_NAME
        _Wrapper.__qualname__ = VLLM_MODEL_NAME
        ModelRegistry.register_model(VLLM_MODEL_NAME, _Wrapper)
    else:  # cross-process string form (SkyRL mp/async workers)
        ModelRegistry.register_model(VLLM_MODEL_NAME, _WRAPPER_IMPORT_PATH)
    logger.info("[zerokl] registered %s into vLLM ModelRegistry", VLLM_MODEL_NAME)
    return VLLM_MODEL_NAME


def find_inprocess_gptmodel(llm):
    """Reach the in-process GPTModelVLLMWrapper inside a vLLM LLM (VLLM_ENABLE_V1_MULTIPROCESSING=0)
    so the trainer can native-sync weights into the rollout model each step."""
    seen = set()
    def walk(o, d=0):
        if id(o) in seen or d > 8:
            return None
        seen.add(id(o))
        if type(o).__name__ == VLLM_MODEL_NAME or hasattr(o, "gpt"):
            return o
        for a in ("llm_engine", "engine_core", "model_executor", "driver_worker",
                  "model_runner", "model", "worker", "engine"):
            if hasattr(o, a):
                try:
                    r = walk(getattr(o, a), d + 1)
                except Exception:
                    r = None
                if r is not None:
                    return r
        return None
    return walk(llm)
