"""Run Qwen3.5-35B-A3B DAPO with native IsoExec models on eight H100s."""

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

from examples.train.isoexec_v2.run import training_overrides


def dapo_overrides(data_dir: Path, run_dir: Path, checkpoint: str | None = None) -> list[str]:
    values = training_overrides("qwen3_5_35b_a3b", data_dir, run_dir, checkpoint=checkpoint)
    settings = {
        "data.train_data": [str(data_dir / "dapo-math-17k-cleaned.parquet")],
        "data.val_data": [str(data_dir / "aime-2024-cleaned.parquet")],
        "trainer.placement.policy_num_gpus_per_node": 8,
        "trainer.placement.ref_num_gpus_per_node": 8,
        "trainer.policy.megatron_config.tensor_model_parallel_size": 4,
        "trainer.policy.megatron_config.expert_model_parallel_size": 8,
        "trainer.policy.megatron_config.expert_tensor_parallel_size": 1,
        "trainer.policy.megatron_config.transformer_config_kwargs": {"sequence_parallel": True},
        "trainer.gradient_checkpointing": False,
        "trainer.algorithm.policy_loss_type": "dual_clip",
        "trainer.algorithm.loss_reduction": "token_mean",
        "trainer.algorithm.eps_clip_low": 0.2,
        "trainer.algorithm.eps_clip_high": 0.28,
        "trainer.algorithm.clip_ratio_c": 10.0,
        "trainer.algorithm.overlong_buffer_len": 4096,
        "trainer.algorithm.overlong_buffer_penalty_factor": 1.0,
        "trainer.algorithm.off_policy_correction.tis_ratio_type": None,
        "trainer.train_batch_size": 128,
        "trainer.policy_mini_batch_size": 32,
        "generator.n_samples_per_prompt": 16,
        "trainer.epochs": 20,
        "trainer.max_training_steps": None,
        "trainer.max_prompt_length": 2048,
        "generator.max_input_length": 2048,
        "trainer.max_tokens_per_microbatch": 20480,
        "trainer.policy.optimizer_config.num_warmup_steps": 40,
        "trainer.logger": "wandb",
        "trainer.project_name": "qwen3.5-35b-dapo",
        "trainer.run_name": "isoexec-native-qwen35-35b-dapo-2048",
        "generator.apply_overlong_filtering": True,
        "generator.sampling_params.max_generate_length": 8192,
        "generator.eval_sampling_params.max_generate_length": 8192,
        "generator.inference_engine.tensor_parallel_size": 8,
        "generator.inference_engine.max_num_batched_tokens": 10240,
        "generator.inference_engine.gpu_memory_utilization": 0.7,
        "generator.inference_engine.engine_init_kwargs": {
            "max_model_len": 10240,
            "max_num_seqs": 512,
            "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
        },
        "environment.env_class": "aime",
    }
    return values + [f"{key}={json.dumps(value)}" for key, value in settings.items()]


def _validate_schedule(overrides: list[str]) -> None:
    schedule = {}
    for item in overrides:
        key, separator, value = item.partition("=")
        if separator and key in ("trainer.max_training_steps", "trainer.policy.optimizer_config.num_warmup_steps"):
            try:
                schedule[key] = json.loads(value)
            except ValueError:
                return  # The native config parser handles other CLI representations.
    steps = schedule.get("trainer.max_training_steps")
    warmup = schedule.get("trainer.policy.optimizer_config.num_warmup_steps")
    if type(steps) is int and type(warmup) is int and steps <= warmup:
        raise ValueError(
            "trainer.max_training_steps must exceed trainer.policy.optimizer_config.num_warmup_steps; "
            "set num_warmup_steps=0 for a short smoke test"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    command = [sys.executable, "-m", "examples.train.algorithms.dapo.main_dapo"]
    overrides = dapo_overrides(args.data_dir, args.run_dir, args.checkpoint) + extra
    try:
        _validate_schedule(overrides)
    except ValueError as exc:
        parser.error(str(exc))
    command += overrides
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.setdefault("SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S", "3600")
    if args.dry_run:
        print(shlex.join(command))
    else:
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
