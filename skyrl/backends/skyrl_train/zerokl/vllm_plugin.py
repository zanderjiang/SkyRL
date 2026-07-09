"""vLLM general plugin: install the zero-KL engine stack in EVERY vLLM process.

WHY. All of the zero-KL engine machinery -- the GPTModel registration, the num_splits=1 CUSTOM varlen
backend, the batch-invariant matmul overrides, the sampler's log_softmax, the GDN autotune/norm pins
-- lives in the process that calls it. At TP=1 with ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the engine
actor IS the worker, so calling them from ``setup_envvars_for_vllm`` was enough, and the whole stack
is documented "Scope: TP=1".

At TP>1 vLLM spawns one worker subprocess per rank. None of them ran any of it. The first symptom is
blunt::

    ValueError: Model architectures ['MegatronGPTModelHybridForCausalLM'] are not supported for now.

and the quieter ones would have been worse: a worker silently using vLLM's default attention backend,
or the fused-Triton sampler, while rank 0 used ours.

``vllm/v1/worker/worker_base.py`` calls ``load_general_plugins()`` in every worker, which runs every
entry point in the ``vllm.general_plugins`` group. This module is that entry point (declared in
``pyproject.toml``). vLLM's own docstring warns plugins may be loaded several times per process --
everything here is idempotent.

Gated on ``SKYRL_ZERO_KL=1``, so a normal SkyRL run is untouched.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_installed = False


def register() -> None:
    """Entry point. Idempotent, and a no-op unless SKYRL_ZERO_KL=1."""
    global _installed

    if _installed or os.environ.get("SKYRL_ZERO_KL") != "1":
        return
    _installed = True

    from .gptmodel_vllm import VLLM_MODEL_NAME, register_gptmodel_to_vllm

    register_gptmodel_to_vllm()

    if os.environ.get("SKYRL_ZEROKL_LOCAL_SPEC") == "1":
        # Registers the @register_backend(CUSTOM) varlen backend. `attention_backend="CUSTOM"` is
        # resolved per worker, so the registration has to exist per worker.
        from . import varlen_backend

        varlen_backend.register_varlen_custom_backend()
        # vLLM's fused-Triton sampler logprob kernel never calls aten log_softmax. The sampler runs
        # in the worker.
        from .vllm_patches import patch_vllm_logprobs_batch_invariant

        patch_vllm_logprobs_batch_invariant()

    if os.environ.get("SKYRL_ZEROKL_MOE_DETERMINISTIC") == "1":
        # SM90: vLLM's batch-invariant mode leaves cuBLAS M-variant. Same override the trainer uses.
        from .moe_batch_invariant import _install_moe_matmul_invariance

        _install_moe_matmul_invariance()

    if os.environ.get("SKYRL_ZEROKL_GDN") == "1":
        # The `fla` facade must exist before megatron.core.ssm.gated_delta_net is imported (the
        # wrapper builds a GPTModel with GDN layers). The zerokl package __init__ already does this
        # on import; call again for clarity and in case of an unusual import order.
        from .gdn_batch_invariant import (
            pin_fla_autotune_configs, pin_gdn_rmsnorm_rows_per_block,
        )
        from .gdn_engine_patch import lift_gdn_batch_invariance_veto
        from .gdn_fla_shim import install_fla_shim

        install_fla_shim()
        pin_fla_autotune_configs()
        pin_gdn_rmsnorm_rows_per_block()
        lift_gdn_batch_invariance_veto()

    print(f"[ZEROKL-PLUGIN] installed in pid {os.getpid()} (arch={VLLM_MODEL_NAME}, "
          f"gdn={os.environ.get('SKYRL_ZEROKL_GDN') == '1'})", flush=True)
