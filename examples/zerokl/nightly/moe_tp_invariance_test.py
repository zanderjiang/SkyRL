"""Is a token's MoE output invariant to the OTHER tokens in the forward, at TP>1?

Zero-KL requires it: the engine pushes a decode batch (few tokens) through the same MoE layer the
trainer pushes a full micro-batch (thousands of tokens) through. If a token's row depends on the
batch, rollout logprobs cannot equal training logprobs -- which is exactly the 7.9e-3 residual seen
on Qwen3.5-35B-A3B (matched TP=8), a model whose MoE runs with expert-tensor-parallelism.

MoE at TP>1 was never validated bitwise: the OLMoE MoE gate ran at TP=1, where the AllGather
dispatcher's dispatch/combine collectives are no-ops. At ETP>1 the combine is a real cross-rank
reduce-scatter over expert-sharded partial sums.

Drives the REAL megatron MoELayer (TopKRouter + AllGather dispatcher + SequentialMLP) under the
full zero-KL recipe. Exit 0 iff the probe row is bitwise-identical across batch compositions.

    CUDA_VISIBLE_DEVICES=0,1 NCCL_ALGO=allreduce:tree NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 \
    uv run --isolated --extra zerokl torchrun --nproc_per_node=2 moe_tp_invariance_test.py
"""
import os
import sys

os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_MOE_DETERMINISTIC", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import skyrl.backends.skyrl_train.zerokl  # noqa: E402,F401  (no-TE guard first)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TP = int(os.environ.get("MOE_TEST_TP", "2"))
E = int(os.environ.get("MOE_TEST_EXPERTS", "8"))
H = 256
FFN = 128


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP, expert_model_parallel_size=1,
        expert_tensor_parallel_size=TP,
    )
    model_parallel_cuda_manual_seed(0)

    from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
    enable_batch_invariant_mode()
    from skyrl.backends.skyrl_train.zerokl.moe_batch_invariant import (
        _install_moe_matmul_invariance, enable_moe_deterministic_ops, lift_moe_tp_sp_training_veto,
    )
    from skyrl.backends.skyrl_train.zerokl.moe_batched_experts import install_batched_sequential_mlp
    _install_moe_matmul_invariance()
    enable_moe_deterministic_ops()
    lift_moe_tp_sp_training_veto()
    install_batched_sequential_mlp()

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

    cfg = TransformerConfig(
        num_layers=1, hidden_size=H, num_attention_heads=4,
        num_moe_experts=E, moe_router_topk=2, ffn_hidden_size=FFN, moe_ffn_hidden_size=FFN,
        moe_grouped_gemm=False, moe_token_dispatcher_type="allgather", moe_router_dtype="fp32",
        gated_linear_unit=True, add_bias_linear=False, activation_func=F.silu,
        tensor_model_parallel_size=TP, expert_model_parallel_size=1,
        expert_tensor_parallel_size=TP, sequence_parallel=False,
        bf16=True, params_dtype=torch.bfloat16,
    )
    spec = get_gpt_layer_local_spec(num_experts=E, moe_grouped_gemm=False)
    moe_partial = spec.submodules.mlp
    pgs = ProcessGroupCollection.use_mpu_process_groups()
    layer = moe_partial(config=cfg, pg_collection=pgs).cuda().eval()

    torch.manual_seed(1234)
    probe = torch.randn(H, device="cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def probe_row(T, seed):
        torch.manual_seed(seed)
        x = torch.randn(T, 1, H, device="cuda", dtype=torch.bfloat16)
        x[0, 0] = probe                     # probe token is always row 0
        out, _ = layer(x)
        return out[0, 0].clone()

    small = probe_row(8, 11)      # engine-like: a tiny decode batch
    large = probe_row(2048, 22)   # trainer-like: a full micro-batch
    other = probe_row(2048, 33)   # same size, different neighbours

    ok_size = torch.equal(small, large)
    ok_content = torch.equal(large, other)
    if rank == 0:
        d1 = (small.float() - large.float()).abs().max().item()
        d2 = (large.float() - other.float()).abs().max().item()
        print(f"[MoE TP={TP} ETP={TP}] probe row invariant to BATCH SIZE   : {ok_size} (maxdiff {d1:.3e})", flush=True)
        print(f"[MoE TP={TP} ETP={TP}] probe row invariant to NEIGHBOURS   : {ok_content} (maxdiff {d2:.3e})", flush=True)
        print("RESULT:", "MoE IS batch-invariant at TP>1" if (ok_size and ok_content)
              else "MoE IS NOT batch-invariant at TP>1 -- this breaks zero-KL", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and not (ok_size and ok_content):
        sys.exit(1)


if __name__ == "__main__":
    main()
