"""Native (no-HF) weight sync for the unified-GPTModel zero-KL route.

When the *generator* (vLLM) runs the SAME Megatron ``GPTModel`` as the trainer, both sides
hold the identical state_dict (same TE-fused ``linear_qkv``/``linear_fc1``, same names and
shapes). So weight sync collapses from the HF round-trip
(``bridge.export_hf_weights`` -> HF layout -> vLLM ``load_weights`` repack) to a **direct
native-layout tensor copy**. This module is that copy.

Dropped vs the HF path: QKV split, gate/up split, vocab gather, HF renames, and vLLM's
HF->fused repack -- plus the seam risks they carried (vocab padding, QKV interleave-vs-concat).
Kept: transport (the caller still moves bytes via CUDA-IPC/NCCL), the bf16 cast, and TP
reshard *only if* degrees differ (native-layout reshard, not a format conversion).

Scope: TP=1 parity path (DTensors are materialized via ``full_tensor``). For TP>1 the reshard
belongs in the transport layer; the names/layout here are unchanged.
"""

from __future__ import annotations

import logging
from typing import Iterator, Tuple

import torch

logger = logging.getLogger(__name__)


def _to_full(t: torch.Tensor) -> torch.Tensor:
    """Materialize a (possibly DTensor) parameter to a plain local tensor."""
    # DTensor (TP/EP sharded) -> full replicated tensor. Plain tensors pass through.
    if hasattr(t, "full_tensor"):
        try:
            return t.full_tensor()
        except Exception:
            return t
    return t


def extract_native_weights(
    actor_module, *, dtype: torch.dtype = torch.bfloat16, include_buffers: bool = False
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(native_name, tensor)`` from a Megatron GPTModel in NATIVE layout.

    No HF conversion. Names are Megatron-native (e.g.
    ``decoder.layers.0.self_attention.linear_qkv.weight``). Tensors are cast to ``dtype``
    (bf16) to match the unified-dtype recipe and the generator's resident precision.

    ``actor_module`` may be a single module or a list (Megatron PP/vpp chunks).
    """
    import os as _os
    _ck = {"s": 0.0, "n": 0}
    modules = actor_module if isinstance(actor_module, (list, tuple)) else [actor_module]
    seen = set()
    for m in modules:
        # Fully unwrap DDP/Float16Module/etc -> bare GPTModel, so names match the engine's
        # `self.gpt` (no `module.` prefix). One unwrap is not enough for DDP(Float16Module(GPTModel)).
        inner = m
        for _ in range(4):
            if hasattr(inner, "module"):
                inner = inner.module
            else:
                break
        for name, p in inner.named_parameters():
            if name in seen:
                continue
            seen.add(name)
            t = _to_full(p.detach()).to(dtype)
            if _os.environ.get("SKYRL_ZERO_KL") == "1":
                _ck["s"] += float(t.float().double().abs().sum()); _ck["n"] += 1
            yield name, t
        if include_buffers:
            for name, b in inner.named_buffers():
                if name in seen or b is None:
                    continue
                seen.add(name)
                yield name, _to_full(b.detach()).to(dtype)
    if _os.environ.get("SKYRL_ZERO_KL") == "1":
        print(f"[ZEROKL-CKSUM] SENDER (trainer) sent {_ck['n']} params, abs-sum checksum={_ck['s']:.6f}", flush=True)


@torch.no_grad()
def param_abs_sum_bf16(modules) -> float:
    """fp64 abs-sum of params on the bf16 basis — the SAME reduction as the weight-sync SENDER
    checksum and the trainer POSTSTEP probe, so forward-time weight states are directly
    comparable across trainer scoring / trainer training / engine forwards. Skips meta params.
    ``modules`` may be a single module or a list; DDP/Float16Module wrappers are unwrapped."""
    mods = modules if isinstance(modules, (list, tuple)) else [modules]
    s = 0.0
    seen = set()
    for m in mods:
        inner = m
        for _ in range(4):
            if hasattr(inner, "module"):
                inner = inner.module
            else:
                break
        for name, p in inner.named_parameters():
            if name in seen or p.device.type == "meta":
                continue
            seen.add(name)
            s += float(p.detach().to(torch.bfloat16).float().double().abs().sum())
    return s


def load_native_weights(target_module, weights_iter, *, strict: bool = True) -> set[str]:
    """Copy ``(native_name, tensor)`` pairs into ``target_module`` in place.

    The target is a GPTModel built with the IDENTICAL spec, so names/shapes match 1:1 and
    this is a straight ``copy_`` -- no repack. Returns the set of loaded names.
    """
    modules = target_module if isinstance(target_module, (list, tuple)) else [target_module]
    dst, dst_bufs = {}, {}
    for m in modules:
        inner = m.module if hasattr(m, "module") else m
        dst.update(dict(inner.named_parameters()))
        dst_bufs.update(dict(inner.named_buffers()))
    loaded = set()
    for name, tensor in weights_iter:
        dest = dst.get(name, dst_bufs.get(name))
        if dest is None:
            if strict:
                raise KeyError(f"[zerokl] native weight name not in target model: {name}")
            continue
        d = _to_full(dest)
        if tuple(d.shape) != tuple(tensor.shape):
            raise ValueError(f"[zerokl] shape mismatch for {name}: {tuple(d.shape)} vs {tuple(tensor.shape)}")
        with torch.no_grad():
            dest.copy_(tensor.to(dest.dtype))
        loaded.add(name)
    missing = set(dst) - loaded
    if strict and missing:
        raise KeyError(f"[zerokl] {len(missing)} target params not synced, e.g. {sorted(missing)[:3]}")
    logger.info("[zerokl] native weight sync: copied %d tensors (no HF conversion)", len(loaded))
    return loaded
