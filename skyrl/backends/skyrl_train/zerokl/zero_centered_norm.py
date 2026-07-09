"""Zero-centered-gamma RMSNorm for the no-TransformerEngine local spec (Qwen3.5 / Qwen3-Next).

Qwen3-Next and Qwen3.5 normalise with ``rms(x) * (1 + w)``: the checkpoint stores gamma centred on
zero, so ``w`` is near 0 rather than near 1. Megatron expresses this as
``config.layernorm_zero_centered_gamma = True`` (megatron-bridge's ``qwen35_bridge`` sets it), and
``TENorm`` implements it. The zero-KL stack has no TransformerEngine, so it uses Megatron's
``WrappedTorchNorm``, which flatly refuses:

    assert not config.layernorm_zero_centered_gamma, "zero_centered_gamma not supported by torch LayerNorm"

That assertion fires while building the very first Qwen3.5 transformer layer, in the trainer and in
the in-vLLM GPTModel alike.

This installs a torch RMSNorm that honours the flag. It delegates the normalisation itself to
``F.rms_norm`` with no weight -- the same aten op the validated dense path already runs through
``torch.nn.RMSNorm``, so it inherits that op's batch invariance -- and applies ``(1 + w)`` afterwards,
which is elementwise and therefore invariant by construction.

Both runtimes install this and both call it, so zero-KL does not depend on it agreeing with TE (it
does not have to; TE is not in the stack). It only has to agree with itself.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_installed = False


class ZeroCenteredTorchRMSNorm(torch.nn.Module):
    """``y = rms_norm(x) * (1 + weight)``. ``weight`` initialised to 0, as the checkpoint expects."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, params_dtype=None):
        super().__init__()
        self.hidden_size = (hidden_size,)
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.zeros(hidden_size, dtype=params_dtype, device=torch.cuda.current_device())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.hidden_size, None, self.eps) * (1.0 + self.weight)

    def extra_repr(self) -> str:
        return f"{self.hidden_size[0]}, eps={self.eps}, zero_centered_gamma=True"


def install_zero_centered_torch_norm() -> bool:
    """Teach Megatron's ``WrappedTorchNorm`` about ``layernorm_zero_centered_gamma``. Idempotent.

    Must run before any transformer layer is built, i.e. alongside the other zerokl import-time
    hooks. Models that do not set the flag get Megatron's original behaviour untouched.
    """
    global _installed

    if _installed:
        return True
    if os.environ.get("SKYRL_ZEROKL_ZERO_CENTERED_NORM", "1") != "1":
        return False
    try:
        from megatron.core.transformer import torch_norm
    except Exception as e:  # pragma: no cover - megatron absent
        logger.info("[zerokl] megatron.core.transformer.torch_norm unavailable (%s)", e)
        return False

    orig_new = torch_norm.WrappedTorchNorm.__new__

    def _new(cls, config, hidden_size, eps=1e-5, **kwargs):
        if not getattr(config, "layernorm_zero_centered_gamma", False):
            return orig_new(cls, config, hidden_size, eps, **kwargs)
        if config.normalization != "RMSNorm":
            raise NotImplementedError(
                f"[zerokl] zero-centered gamma is only implemented for RMSNorm, got {config.normalization}"
            )
        assert not config.sequence_parallel, "sequence parallel not supported by torch LayerNorm"
        return ZeroCenteredTorchRMSNorm(hidden_size, eps, params_dtype=config.params_dtype)

    torch_norm.WrappedTorchNorm.__new__ = _new
    _installed = True
    print("[ZEROKL] WrappedTorchNorm now supports layernorm_zero_centered_gamma "
          "(rms_norm(x) * (1 + w)) -- required by Qwen3.5 / Qwen3-Next", flush=True)
    return True
