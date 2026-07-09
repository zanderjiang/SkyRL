"""TP-matched TRAINER scorer: score a dumped engine rollout with a Megatron-TP GPTModel.

The TP>1 counterpart of engine_trainer_parity_harness.py's (C) phase. skyrl_engine_parity_test.py
(PARITY_TP=N PARITY_DUMP=<pt>) writes {prompt_ids, gen_ids, decode_lps, prefill_lps}; this script
builds the TRAINER-side GPTModel exactly as megatron_worker does (local spec + GDN shim + varlen
num_splits=1 + batch-invariant aten + scoring_mode), sharded over Megatron TP == torchrun world,
scores the same full sequence, extracts logprobs through the REAL from_parallel_logits_to_logprobs
(which under SKYRL_ZERO_KL=1 gathers the vocab shards and applies the engine's exact lse formula),
and prints per-token trainer-vs-engine diffs.

Launch:
    SKYRL_ZERO_KL=1 SKYRL_ZEROKL_LOCAL_SPEC=1 SKYRL_ZEROKL_GDN=1 VLLM_BATCH_INVARIANT=1 \
    VARLEN_FORCE_NUM_SPLITS_1=1 HF_HOME=/mnt/local_storage/hf \
    ZEROKL_MODEL=Qwen/Qwen3.5-0.8B ZK_DUMP=/mnt/local_storage/logs/parity_tp2_rollout.pt \
    CUDA_VISIBLE_DEVICES=0,1 NCCL_ALGO=allreduce:tree NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 \
    uv run --isolated --extra zerokl torchrun --nproc_per_node=2 trainer_score_tp.py
"""
import os

os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_GDN", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VARLEN_FORCE_NUM_SPLITS_1", "1")

MODEL = os.environ.get("ZEROKL_MODEL", "Qwen/Qwen3.5-0.8B")
DUMP = os.environ.get("ZK_DUMP", "/mnt/local_storage/logs/parity_tp2_rollout.pt")

# zerokl package first: no-TE guard + fla shim before megatron imports.
import skyrl.backends.skyrl_train.zerokl  # noqa: E402,F401
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def build_trainer_gpt(tp: int):
    """Build the trainer GPTModel the way megatron_worker.init_configs does (Qwen3.5 GDN path)."""
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend
    from transformers import AutoConfig
    from skyrl.backends.skyrl_train.zerokl import make_zerokl_local_layer_spec
    from skyrl.backends.skyrl_train.workers.megatron.model_bridges import (
        maybe_force_qwen35_text_bridge,
    )
    from skyrl.backends.skyrl_train.zerokl.gdn_hybrid_spec import (
        checkpoint_is_vl_named, patch_qwen35_bridge_for_local_spec,
    )

    b = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
    vl = checkpoint_is_vl_named(b.hf_pretrained.config)
    patch_qwen35_bridge_for_local_spec(hf_lm_prefix="model.language_model." if vl else None)
    if maybe_force_qwen35_text_bridge(b, b.hf_pretrained.config):
        print("[TRAINER-TP] forced Qwen3.5 TEXT bridge", flush=True)
    mp = b.to_megatron_provider(load_weights=True)
    mp.tensor_model_parallel_size = tp
    mp.pipeline_model_parallel_size = 1
    mp.expert_model_parallel_size = 1
    mp.expert_tensor_parallel_size = tp
    mp.sequence_parallel = False
    mp.pipeline_dtype = torch.bfloat16
    mp.apply_rope_fusion = False
    mp.attention_backend = AttnBackend.flash
    mp.gradient_accumulation_fusion = False
    mp.transformer_layer_spec = make_zerokl_local_layer_spec(mp)
    if getattr(mp, "mtp_num_layers", None):
        mp.mtp_num_layers = None
    hf = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    hft = getattr(hf, "text_config", hf)
    rp = getattr(hft, "rope_parameters", None) or getattr(hft, "rope_scaling", None)
    if isinstance(rp, dict) and rp.get("rope_theta"):
        mp.rotary_base = rp["rope_theta"]
    elif getattr(hft, "rope_theta", None):
        mp.rotary_base = hft.rope_theta
    mp.finalize()
    gpt_list = mp.provide_distributed_model(wrap_with_ddp=False)
    bare = gpt_list[0]
    for _ in range(4):
        if hasattr(bare, "decoder"):
            break
        bare = getattr(bare, "module", bare)
    print(f"[TRAINER-TP] GPTModel built (rotary_base={getattr(mp, 'rotary_base', '?')}) "
          f"params={sum(1 for _ in bare.named_parameters())}", flush=True)
    return bare


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    mpu.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(0)

    bare = build_trainer_gpt(world)
    bare.eval()

    # the worker's trainer-side kernel patches (megatron_worker init_model order)
    from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import (
        enable_trainer_batch_invariant, swap_trainer_core_attention_varlen,
    )
    enable_trainer_batch_invariant()
    swap_trainer_core_attention_varlen(bare)

    d = torch.load(DUMP)
    full = list(d["prompt_ids"]) + list(d["gen_ids"])
    P, n, L = len(d["prompt_ids"]), len(d["gen_ids"]), len(full)
    dev = torch.device(f"cuda:{rank}")
    seq = torch.tensor(full, dtype=torch.long, device=dev).unsqueeze(0)
    pos = torch.arange(L, device=dev).unsqueeze(0)

    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        from_parallel_logits_to_logprobs,
    )
    from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
        _zerokl_scoring_ctx,
    )

    with torch.no_grad(), _zerokl_scoring_ctx():
        logits = bare(input_ids=seq, position_ids=pos, attention_mask=None)  # [1, L, V/TP]
        tp_rank = mpu.get_tensor_model_parallel_rank()
        shard = logits.shape[-1]
        lp = from_parallel_logits_to_logprobs(
            logits, seq,
            vocab_start_index=tp_rank * shard,
            vocab_end_index=(tp_rank + 1) * shard,
            tp_group=mpu.get_tensor_model_parallel_group(),
            inference_only=True, cp_group=None, chunk_size=None,
        )  # [1, L-1]; lp[0, t] = logprob of full[t+1]

    if rank == 0:
        trainer_lps = lp[0, P - 1: P + n - 1].float().cpu()
        pre = torch.tensor(d["prefill_lps"])
        dec = torch.tensor(d["decode_lps"])
        d_pre = (trainer_lps - pre).abs()
        d_dec = (trainer_lps - dec).abs()
        print(f"\n[C] trainer vs engine PREFILL: max={float(d_pre.max()):.6f} mean={float(d_pre.mean()):.7f} "
              f"exact0={int((d_pre == 0).sum())}/{n}", flush=True)
        print(f"[C] trainer vs engine DECODE : max={float(d_dec.max()):.6f} mean={float(d_dec.mean()):.7f} "
              f"exact0={int((d_dec == 0).sum())}/{n}", flush=True)
        top = torch.topk(d_pre, k=min(10, n)).indices.tolist()
        print("[C] top trainer-vs-prefill outliers:", flush=True)
        for i in top:
            print(f"   i={i}/{n} abs_pos={P+i} d_pre={float(d_pre[i]):.5f} d_dec={float(d_dec[i]):.5f} "
                  f"trn={float(trainer_lps[i]):.5f} pre={float(pre[i]):.5f}", flush=True)
        print("\nRESULT:", "TRAINER==ENGINE-PREFILL BITWISE" if float(d_pre.max()) == 0
              else f"RESIDUAL max={float(d_pre.max()):.4f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
