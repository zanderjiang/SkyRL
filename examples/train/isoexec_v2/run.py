"""Run GSM8K with IsoExec Model V2, Megatron training and vLLM inference."""

import argparse
import json
import os
from pathlib import Path
import shlex
import sys


MODELS = {
    "qwen3_5_35b_a3b": (
        "Qwen/Qwen3.5-35B-A3B-Base",
        "isoexec.models.v2.qwen3_5_moe",
        None,
    ),
    "qwen3_5_0_8b": (
        "Qwen/Qwen3.5-0.8B",
        "isoexec.models.v2.qwen3_5",
        "2fc06364715b967f1860aea9cf38778875588b17",
    ),
    "qwen3_5_9b": (
        "Qwen/Qwen3.5-9B",
        "isoexec.models.v2.qwen3_5",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    ),
    "qwen1_5_moe_a2_7b": (
        "Qwen/Qwen1.5-MoE-A2.7B",
        "isoexec.models.v2.qwen2_moe",
        "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
    ),
}


def training_overrides(
    model: str,
    data_dir: Path,
    run_dir: Path,
    *,
    checkpoint: str | None = None,
    revision: str | None = None,
    steps: int = 1,
) -> list[str]:
    """Keep model geometry in the selected checkpoint's HF config.json."""
    hf_repo, package, default_revision = MODELS[model]
    # IsoExec Model V2: import the shared architecture; read dimensions from HF config.json.
    source: dict[str, object] = {
        "source": {"api_version": 2, "package": package},
        # IsoExec placement policy for trainer and inference parameters.
        "trainer_placement": {"leaves": 8, "leaf_dtype": "bf16"},
        "inference_placement": {"leaves": 8, "leaf_dtype": "bf16"},
    }
    selected_revision = revision if checkpoint is not None else revision or default_revision
    if selected_revision is not None:
        source["revision"] = selected_revision
    values: dict[str, object] = {
        "data.train_data": [str(data_dir / "train.parquet")],
        "data.val_data": [str(data_dir / "validation.parquet")],
        # IsoExec opt-in and the worker-serializable ModelSource descriptor.
        "enable_isoexec": True,
        "isoexec_model": source,
        "trainer.strategy": "megatron",
        "trainer.policy.model.path": checkpoint or hf_repo,
        "trainer.policy.language_model_only": True,
        "trainer.ref.language_model_only": True,
        "trainer.policy.use_torch_compile": False,
        "trainer.bf16": True,
        "trainer.remove_microbatch_padding": True,
        "trainer.gradient_checkpointing": False,
        "trainer.mtp.enabled": False,
        "trainer.fused_lm_head_logprob": False,
        # IsoExec compares every vocabulary entry at rollout/prescore token positions.
        "trainer.rollout_logprob_comparison": "full",
        "trainer.recompute_old_logprobs_per_minibatch": True,
        "trainer.placement.colocate_all": True,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.policy_num_gpus_per_node": 1,
        "trainer.placement.ref_num_nodes": 1,
        "trainer.placement.ref_num_gpus_per_node": 1,
        "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
        "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
        "trainer.policy.megatron_config.context_parallel_size": 1,
        "trainer.policy.megatron_config.expert_model_parallel_size": 1,
        "trainer.policy.megatron_config.expert_tensor_parallel_size": 1,
        "trainer.policy.megatron_config.transformer_config_kwargs": {
            "sequence_parallel": False,
        },
        "trainer.policy.megatron_config.optimizer_config_kwargs": {
            "optimizer_cpu_offload": True,
            "optimizer_offload_fraction": 1.0,
            "use_torch_optimizer_for_cpu_offload": True,
            "overlap_cpu_optimizer_d2h_h2d": False,
            "use_precision_aware_optimizer": True,
            "pin_cpu_grads": False,
            "pin_cpu_params": False,
        },
        "trainer.policy.megatron_config.moe_router_load_balancing_type": (
            "aux_loss" if model == "qwen1_5_moe_a2_7b" else "none"
        ),
        "trainer.policy.megatron_config.moe_aux_loss_coeff": 0.001 if model == "qwen1_5_moe_a2_7b" else 0.0,
        "trainer.epochs": 1,
        "trainer.max_training_steps": steps,
        "trainer.update_epochs_per_batch": 1,
        "trainer.train_batch_size": 4,
        "trainer.policy_mini_batch_size": 4,
        "trainer.micro_forward_batch_size_per_gpu": 1,
        "trainer.micro_train_batch_size_per_gpu": 1,
        "trainer.max_tokens_per_microbatch": -1,
        "trainer.max_prompt_length": 512,
        "trainer.eval_before_train": False,
        "trainer.eval_interval": -1,
        "trainer.ckpt_interval": -1,
        "trainer.hf_save_interval": -1,
        "trainer.resume_mode": None,
        "trainer.algorithm.advantage_estimator": "grpo",
        "trainer.algorithm.use_kl_loss": False,
        "trainer.algorithm.dynamic_sampling.type": None,
        "trainer.policy.optimizer_config.lr": 1e-6,
        "trainer.policy.optimizer_config.num_warmup_steps": 0,
        "trainer.logger": "console",
        "trainer.project_name": "isoexec-model-v2",
        "trainer.run_name": model,
        "trainer.log_path": str(run_dir / "logs"),
        "trainer.ckpt_path": str(run_dir / "checkpoints"),
        "trainer.export_path": str(run_dir / "exports"),
        "generator.batched": True,
        "generator.n_samples_per_prompt": 5,
        "generator.max_turns": 1,
        "generator.sampling_params.max_generate_length": 1024,
        "generator.sampling_params.min_tokens": 0,
        "generator.inference_engine.backend": "vllm",
        "generator.inference_engine.language_model_only": True,
        "generator.inference_engine.run_engines_locally": True,
        "generator.inference_engine.num_engines": 1,
        "generator.inference_engine.tensor_parallel_size": 1,
        "generator.inference_engine.pipeline_parallel_size": 1,
        "generator.inference_engine.data_parallel_size": 1,
        "generator.inference_engine.expert_parallel_size": 1,
        "generator.inference_engine.weight_sync_backend": "nccl",
        # Must match trainer.rollout_logprob_comparison for IsoExec full-row comparison.
        "generator.inference_engine.logprob_output": "full",
        "generator.inference_engine.enable_prefix_caching": False,
        "generator.inference_engine.enforce_eager": False,
        "generator.inference_engine.gpu_memory_utilization": 0.5,
        "generator.inference_engine.max_num_batched_tokens": 2048,
        "generator.inference_engine.engine_init_kwargs": {
            "max_model_len": 2048,
            "max_num_seqs": 20,
            # IsoExec keeps its OP bindings; capture decode without torch.compile.
            "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
        },
        "environment.env_class": "gsm8k",
    }
    return [f"{key}={json.dumps(value)}" for key, value in values.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="qwen3_5_0_8b")
    parser.add_argument("--checkpoint", help="Local/private checkpoint directory or a Hub model ID")
    parser.add_argument("--revision", help="Hub revision; built-in model IDs default to pinned commits")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "data/gsm8k")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/isoexec-v2"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting Ray or a GPU")
    args, extra = parser.parse_known_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    command = [sys.executable, "-m", "skyrl.train.entrypoints.main_base"]
    command += training_overrides(
        args.model, args.data_dir, args.run_dir, checkpoint=args.checkpoint, revision=args.revision, steps=args.steps
    )
    command += extra
    # IsoExec's prepared native SDK must also be used by Ray workers.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    # IsoExec's first native kernel builds can exceed SkyRL's default startup window.
    os.environ.setdefault("SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S", "1200")
    if args.dry_run:
        print(
            shlex.join(
                [
                    "env",
                    "RAY_ENABLE_UV_RUN_RUNTIME_ENV=" + os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"],
                    "SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S="
                    + os.environ["SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S"],
                    *command,
                ]
            )
        )
        return
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
