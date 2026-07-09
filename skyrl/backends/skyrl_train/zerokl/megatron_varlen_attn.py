"""Grad-capable trainer attention that is BITWISE-identical to the zero-KL engine kernel.

The rollout engine (gptmodel_vllm + zerokl/varlen_backend) computes attention with
`torch.nn.attention.varlen.varlen_attn(..., num_splits=1, window_size=(-1, 0))` -- the single-split,
causal (unlimited-left / zero-right window) PyTorch varlen FlashAttention. To drive
`minibatch_rollout_logprobs_abs_diff` to EXACTLY 0 (not ~1e-3 with huge per-token outliers), the
Megatron TRAINER's `SelfAttention.core_attention` must call the SAME function with the SAME args.
torch SDPA (the local-spec default) and `flash_attn` (the production swap) are DIFFERENT kernels and
leave occasional catastrophic per-token logprob outliers (max ~10-17) -- that is the "KL is very high"
symptom even though the mean is tiny.

Verified: `varlen_attn(..., window_size=(-1,0))` matches SDPA-causal (causality correct) and supports
autograd. b==1 per micro-forward (micro_*_batch_size_per_gpu=1); right-padding after the real tokens is
harmless under causal attention.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class TorchVarlenCoreAttn(nn.Module):
    """Drop-in for ``SelfAttention.core_attention`` using torch ``varlen_attn`` == the engine kernel."""

    def __init__(self, *, num_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        import os as _os
        import torch.nn.attention.varlen as _V  # noqa: N814

        self._varlen_attn = _V.varlen_attn
        # varlen_attn (non-paged) does NOT reliably honor num_splits=1 across runtime contexts -- in the
        # distributed training forward (different GPU occupancy/streams) its FA3 split-K heuristic picks a
        # different reduction than a clean process, giving a 1-ULP bf16 core_attention diff that amplifies
        # into the ~0.014 rollout-vs-train gap (localized via trace_layerwise.py: FIRST divergence at
        # decoder.layers.0.self_attention.core_attention, inputs bitwise). varlen_attn_out (the ENGINE's
        # kernel, varlen_backend.py) DOES honor num_splits=1 (proven: engine decode==prefill bitwise across
        # contexts). Default to it so the trainer attention is context-invariant == the engine.
        self._use_out = _os.environ.get("SKYRL_ZEROKL_VARLEN_OUT", "0") == "1"
        self._varlen_attn_out = getattr(_V, "varlen_attn_out", None)
        if self._use_out and self._varlen_attn_out is None:
            self._use_out = False
        # PAGED path (block_table) forces num_splits=1 to be honored context-invariantly -- the ONLY
        # variant that matches the engine's decode==prefill bitwise across runtime contexts. The non-paged
        # varlen_attn/varlen_attn_out pick a context-dependent split-K in the distributed forward (localized
        # as the FIRST divergence at core_attention). block_size must be divisible by 256 (FA3 constraint).
        self._paged = _os.environ.get("SKYRL_ZEROKL_VARLEN_PAGED", "0") == "1" and self._varlen_attn_out is not None
        self._page_bs = 256
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.enable_gqa = num_heads > num_kv_heads

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        # Two layouts reach this kernel:
        #   sbhd  q [sq, b, np, hn] -- the unpacked micro-forward, b == 1.
        #   thd   q [t,  np, hn]    -- sample packing; Megatron folds the batch dim away and passes
        #                             the sequence boundaries in `packed_seq_params`.
        if query.dim() == 4:
            sq, b = query.shape[0], query.shape[1]
            assert b == 1, "TorchVarlenCoreAttn supports the b=1 micro-forward (micro_*_batch_size_per_gpu=1)"
            packed = False
        elif query.dim() == 3:
            sq, b = query.shape[0], 1
            packed = True
        else:
            raise ValueError(f"unexpected core_attention query rank {query.dim()}")
        q = query.reshape(sq, self.num_heads, self.head_dim)
        k = key.reshape(sq, self.num_kv_heads, self.head_dim)
        v = value.reshape(sq, self.num_kv_heads, self.head_dim)

        # WITHOUT the packed boundaries a packed row would attend straight across sequence
        # boundaries -- causal within the row, but a token of sequence 2 would see sequence 1. The
        # rollout engine attends per sequence, so that alone would break zero-KL.
        max_q = max_k = sq
        if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd":
            cu = packed_seq_params.cu_seqlens_q.to(device=q.device, dtype=torch.int32)
            cu_kv = packed_seq_params.cu_seqlens_kv.to(device=q.device, dtype=torch.int32)
            if not torch.equal(cu, cu_kv):
                raise NotImplementedError("[zerokl] cu_seqlens_q != cu_seqlens_kv")
            max_q = int(packed_seq_params.max_seqlen_q)
            max_k = int(packed_seq_params.max_seqlen_kv)
        elif packed:
            raise ValueError("[zerokl] thd core_attention input without PackedSeqParams(qkv_format='thd')")
        else:
            cu = torch.tensor([0, sq], device=q.device, dtype=torch.int32)
        if self._paged:
            if packed:
                raise NotImplementedError("[zerokl] the PAGED varlen recipe does not handle thd packing")
            # engine's PAGED recipe: pack K/V into [num_blocks, 256, kv_heads, head_dim] + block_table so
            # FA3 honors num_splits=1 (context-invariant). Non-inplace pad keeps autograd intact.
            _bs = self._page_bs
            nb = (sq + _bs - 1) // _bs
            pad = nb * _bs - sq
            if pad:
                k = torch.cat([k, k.new_zeros(pad, self.num_kv_heads, self.head_dim)], dim=0)
                v = torch.cat([v, v.new_zeros(pad, self.num_kv_heads, self.head_dim)], dim=0)
            kc = k.view(nb, _bs, self.num_kv_heads, self.head_dim)
            vc = v.view(nb, _bs, self.num_kv_heads, self.head_dim)
            bt = torch.arange(nb, device=q.device, dtype=torch.int32).unsqueeze(0)
            su = torch.tensor([sq], device=q.device, dtype=torch.int32)
            _o = torch.empty_like(q)
            out = self._varlen_attn_out(
                _o, q, kc, vc, cu, cu, sq, sq,
                scale=self.scale, num_splits=1, enable_gqa=self.enable_gqa, window_size=(-1, 0),
                block_table=bt, seqused_k=su,
            )
        elif self._use_out:
            # engine's exact kernel (varlen_attn_out) -> honors num_splits=1 context-invariantly.
            _o = torch.empty_like(q)
            out = self._varlen_attn_out(
                _o, q, k, v, cu, cu, max_q, max_k,
                scale=self.scale, num_splits=1, enable_gqa=self.enable_gqa, window_size=(-1, 0),
            )
        else:
            out = self._varlen_attn(
                q, k, v, cu, cu, max_q, max_k,
                scale=self.scale,
                num_splits=1,             # single KV-reduction split -> bitwise == engine prefill/decode
                enable_gqa=self.enable_gqa,
                window_size=(-1, 0),      # unlimited left, zero right == causal (the engine's recipe)
            )
        if isinstance(out, tuple):
            out = out[0]
        hp = self.num_heads * self.head_dim
        # thd keeps the folded layout Megatron handed us; sbhd gets its batch dim back.
        return out.reshape(sq, hp) if packed else out.reshape(sq, b, hp)


def enable_trainer_batch_invariant():
    """Enable the SAME vLLM batch-invariant aten ops the engine runs under VLLM_BATCH_INVARIANT
    (mm/addmm/matmul/linear/_log_softmax/mean.dim/rms_norm), so the trainer's NON-attention ops are
    bitwise-identical to the rollout. Without this the trainer GEMM/RMSNorm/logits use ordinary
    (batch-variant) kernels and leave a small residual even after the attention kernel is matched.
    Uses vLLM's implementation (not megatron-core's) so trainer and engine share the exact same kernels.
    Idempotent (vLLM guards with a module-global flag)."""
    try:
        from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
    except Exception as e:  # pragma: no cover
        logger.warning("[zerokl] vLLM batch_invariant unavailable, trainer non-attn not batch-invariant: %s", e)
        return False
    enable_batch_invariant_mode()
    print("[ZEROKL-TRAINER] enabled vLLM batch-invariant aten ops (mm/addmm/linear/log_softmax/mean) "
          "== engine -> bitwise non-attention", flush=True)
    return True


def activate_trainer_flash_attention_impl():
    """Activate torch's FA3 flash-attention impl in the TRAINER runtime == the engine does.

    ROOT CAUSE of the ~0.014 zero-KL residual (verified probe_attn_variants.py on the real 2239-tok
    rollout: baseline mean 0.01386 / max 0.2976, after activation bitwise 0/2048): the engine's varlen
    backend calls ``activate_flash_attention_impl("FA3")`` (varlen_backend.py) so its
    ``varlen_attn(num_splits=1, window=(-1,0))`` dispatches to FA3. The trainer process builds NO vLLM
    engine, so torch's flash impl stays UNSET (``current_flash_attention_impl()`` is ``None``) and the
    SAME ``varlen_attn`` call dispatches to a DIFFERENT (non-FA3) kernel -> a 1-ULP bf16 core_attention
    diff that amplifies into the ~0.014 rollout-vs-train residual on real temp-1.0 rollouts. Every
    in-process diagnostic harness that built a vLLM engine silently activated FA3, which is why they all
    measured bitwise while the LIVE distributed trainer diverged. Activating FA3 here (same guard as the
    engine's ``_has_sm90()``) makes the trainer's plain non-paged ``varlen_attn`` bitwise == the engine.
    """
    try:
        from torch.nn.attention import activate_flash_attention_impl, current_flash_attention_impl
    except Exception as e:  # pragma: no cover - old torch without the flash-impl API
        logger.warning("[zerokl] torch flash-attn impl API unavailable, trainer attn may not match engine: %s", e)
        return None
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        logger.warning("[zerokl] FA3 requires SM 9.0+; trainer attention will NOT be bitwise == engine")
        return current_flash_attention_impl()
    if current_flash_attention_impl() != "FA3":
        activate_flash_attention_impl("FA3")
    impl = current_flash_attention_impl()
    print(f"[ZEROKL-TRAINER] activated flash-attn impl = {impl} (== engine varlen_backend FA3) "
          f"-> trainer varlen_attn bitwise == engine", flush=True)
    return impl


def swap_trainer_core_attention_varlen(gpt_modules):
    """Replace each decoder layer's core_attention with the torch-varlen kernel (== rollout engine)."""
    # Match the engine's flash-attn dispatch BEFORE any trainer forward -- this is the actual zero-KL fix
    # (the varlen kernel choice/paging was a red herring; FA3-vs-not was the ~0.014 residual).
    activate_trainer_flash_attention_impl()
    modules = gpt_modules if isinstance(gpt_modules, (list, tuple)) else [gpt_modules]
    n = 0
    for m in modules:
        inner = m
        for _ in range(4):  # unwrap DDP(Float16Module(GPTModel)) -> GPTModel (the one with .decoder)
            if hasattr(inner, "decoder"):
                break
            inner = getattr(inner, "module", inner)
        if not hasattr(inner, "decoder"):
            continue
        cfg = inner.config
        head_dim = getattr(cfg, "kv_channels", cfg.hidden_size // cfg.num_attention_heads)
        for layer in inner.decoder.layers:
            sa = getattr(layer, "self_attention", None)
            if sa is None:
                continue
            sa.core_attention = TorchVarlenCoreAttn(
                num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_query_groups,
                head_dim=head_dim, scale=head_dim ** -0.5)
            n += 1
    import os as _os2
    _mode = ("PAGED varlen_attn_out" if _os2.environ.get("SKYRL_ZEROKL_VARLEN_PAGED") == "1"
             else "varlen_attn_out (non-paged)" if _os2.environ.get("SKYRL_ZEROKL_VARLEN_OUT") == "1"
             else "varlen_attn (non-paged)")
    logger.info("[zerokl] swapped TRAINER core_attention -> %s on %d layers", _mode, n)
    print(f"[ZEROKL-TRAINER] swapped core_attention -> {_mode} num_splits=1 window=(-1,0) on {n} layers "
          f"(VARLEN_PAGED={_os2.environ.get('SKYRL_ZEROKL_VARLEN_PAGED')!r} "
          f"VARLEN_OUT={_os2.environ.get('SKYRL_ZEROKL_VARLEN_OUT')!r})", flush=True)
    return n
