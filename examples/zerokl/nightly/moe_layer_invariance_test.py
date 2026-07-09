"""Bitwise batch-invariance of a Megatron MoE layer under the zero-KL recipe. Single GPU, no Ray.

Gate 1c. The zero-KL invariant, applied to the MoE layer: a token's output must be identical
whether it was computed inside a full 512-token prefill or as a lone token in an incremental decode
step. Two megatron-core ops break that (see zerokl/moe_batch_invariant.py for the audit):

  * the expert combine is a CUDA ``scatter_add_`` with duplicate indices -> atomicAdd -> the top-k
    expert outputs are summed in hardware-arbitrary order;
  * the router's ``torch.topk(..., sorted=torch.is_grad_enabled())`` returns the top-k in a
    different order under grad than under no_grad, and with ``moe_router_pre_softmax=False`` that
    order feeds a softmax reduction -> the engine (no_grad) and trainer (grad) disagree.

This test measures both, first unpatched (expect nonzero) and then patched (require max == 0.0).

Run on the zero-KL nightly venv, one free GPU:
    SKYRL_ZEROKL_LOCAL_SPEC=1 VLLM_BATCH_INVARIANT=1 CUDA_VISIBLE_DEVICES=<gpu> \
    uv run --isolated --extra zerokl python examples/zerokl/nightly/moe_layer_invariance_test.py \
      > /mnt/local_storage/logs/moe_layer_invariance.log 2>&1

Env knobs: MOE_SEQLEN (512), MOE_EXPERTS (64), MOE_TOPK (8), MOE_HIDDEN (512), MOE_FFN (256).
"""

import os

os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29571")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SEQLEN = int(os.environ.get("MOE_SEQLEN", "512"))
NUM_EXPERTS = int(os.environ.get("MOE_EXPERTS", "64"))
TOPK = int(os.environ.get("MOE_TOPK", "8"))
HIDDEN = int(os.environ.get("MOE_HIDDEN", "512"))
FFN = int(os.environ.get("MOE_FFN", "256"))


def _init_single_gpu_megatron():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
    model_parallel_cuda_manual_seed(1234)


def build_moe_layer(*, pre_softmax: bool):
    """A single MoELayer built exactly as the zero-KL local spec builds it."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        num_attention_heads=8,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_ffn_hidden_size=FFN,
        moe_router_pre_softmax=pre_softmax,
        moe_router_score_function="softmax",
        # the zero-KL MoE pin (force_zerokl_moe_config applies exactly these on both sides)
        moe_grouped_gemm=False,
        moe_token_dispatcher_type="allgather",
        moe_router_dtype="fp32",
        moe_permute_fusion=False,
        moe_router_fusion=False,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
    )
    spec = get_gpt_layer_local_spec(
        num_experts=NUM_EXPERTS, moe_grouped_gemm=False, qk_layernorm=False, normalization="RMSNorm"
    )
    layer = spec.submodules.mlp(config=config, layer_number=1).cuda()
    layer.eval()
    return layer


@torch.no_grad()
def decode_vs_prefill(layer, x):
    """max |full-sequence output[i] - single-token output for token i| over all i."""
    full, _ = layer(x)
    worst = 0.0
    for i in range(x.shape[0]):
        one, _ = layer(x[i : i + 1])
        worst = max(worst, float((full[i] - one[0]).abs().max()))
    return worst


def grad_vs_nograd(layer, x):
    """max |output under enable_grad - output under no_grad| on the same full sequence.

    The engine always runs no_grad; the trainer's training forward runs under grad. Any op keyed on
    ``torch.is_grad_enabled()`` (the router's top-k ``sorted`` flag) shows up here.
    """
    with torch.no_grad():
        a, _ = layer(x)
    with torch.enable_grad():
        b, _ = layer(x)
    return float((a - b.detach()).abs().max())


def main():
    from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode

    from skyrl.backends.skyrl_train.zerokl.moe_batch_invariant import (
        enable_moe_deterministic_ops,
        revert_moe_deterministic_ops,
    )

    _init_single_gpu_megatron()
    enable_batch_invariant_mode()  # the same aten overrides the engine and trainer run under
    print(
        f"=== MoE layer batch-invariance | torch {torch.__version__} | seq={SEQLEN} "
        f"experts={NUM_EXPERTS} topk={TOPK} hidden={HIDDEN} ffn={FFN} ===",
        flush=True,
    )

    torch.manual_seed(0)
    x = torch.randn(SEQLEN, 1, HIDDEN, dtype=torch.bfloat16, device="cuda")

    results = {}
    for pre_softmax in (True, False):
        tag = "pre_softmax" if pre_softmax else "post_softmax"
        layer = build_moe_layer(pre_softmax=pre_softmax)

        revert_moe_deterministic_ops()
        base_dp = decode_vs_prefill(layer, x)
        base_gn = grad_vs_nograd(layer, x)

        enable_moe_deterministic_ops()
        fixed_dp = decode_vs_prefill(layer, x)
        fixed_gn = grad_vs_nograd(layer, x)
        revert_moe_deterministic_ops()

        results[tag] = (base_dp, base_gn, fixed_dp, fixed_gn)
        print(
            f"\n[{tag}] moe_router_pre_softmax={pre_softmax}\n"
            f"  decode-vs-prefill  unpatched max={base_dp:.6e}   patched max={fixed_dp:.6e}\n"
            f"  grad-vs-nograd     unpatched max={base_gn:.6e}   patched max={fixed_gn:.6e}",
            flush=True,
        )

    failures = [
        f"{tag}: decode_vs_prefill={dp:.3e} grad_vs_nograd={gn:.3e}"
        for tag, (_, _, dp, gn) in results.items()
        if dp != 0.0 or gn != 0.0
    ]
    if failures:
        print("\nRESULT: FAIL (patched path is not bitwise)\n  " + "\n  ".join(failures), flush=True)
        raise SystemExit(1)
    print("\nRESULT: BITWISE-IDENTICAL (max == 0.0 for every configuration, patched)", flush=True)


if __name__ == "__main__":
    main()
