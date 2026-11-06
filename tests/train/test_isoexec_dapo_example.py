"""The production recipe keeps native placement and rollout sizing explicit."""

import json
from pathlib import Path

from examples.train.isoexec.run_dapo import dapo_overrides

import pytest

# run_dapo.main() sets these with os.environ.setdefault; restore them so later tests
# (for example the GSM8K launcher's own defaults) see the process environment unchanged.
_LAUNCHER_ENV = (
    "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
    "SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S",
    "ISOEXEC_MOE_AFTER_FC2_OWNER",
    "ISOEXEC_MOE_AFTER_FC2_GATHER",
)


@pytest.fixture(autouse=True)
def _restore_launcher_environment(monkeypatch):
    for name in _LAUNCHER_ENV:
        monkeypatch.setenv(name, "")  # records the original value (or absence) for undo
        monkeypatch.delenv(name)


def test_dapo_recipe_has_2048_rollouts_and_activation_checkpointing():
    values = {
        key: json.loads(value)
        for item in dapo_overrides(Path("/data"), Path("/run"))
        for key, value in [item.split("=", 1)]
    }
    assert values["enable_isoexec"] is True
    assert values["isoexec_model"]["source"]["package"] == "isoexec.models.qwen3_5_moe"
    assert values["trainer.train_batch_size"] * values["generator.n_samples_per_prompt"] == 2048
    assert values["trainer.gradient_checkpointing"] is True
    assert values["generator.sampling_params.logprobs"] == 0
    assert values["trainer.ckpt_interval"] == values["trainer.hf_save_interval"] == -1
    assert values["trainer.policy.megatron_config.tensor_model_parallel_size"] == 4
    assert values["trainer.policy.megatron_config.expert_model_parallel_size"] == 8
    assert values["generator.inference_engine.tensor_parallel_size"] == 8
    assert values["trainer.policy.megatron_config.transformer_config_kwargs"] == {"sequence_parallel": True}


def test_smoke_overrides_reach_dry_run(monkeypatch, capsys):
    import shlex
    import sys

    from examples.train.isoexec.run_dapo import main

    overrides = [
        "trainer.max_training_steps=2",
        "trainer.policy.optimizer_config.num_warmup_steps=0",
        "trainer.rollout_logprob_comparison=action",
        "generator.inference_engine.logprob_output=action",
        "generator.sampling_params.max_generate_length=4096",
        "trainer.algorithm.overlong_buffer_len=1024",
        "generator.eval_sampling_params.max_generate_length=4096",
        "trainer.max_tokens_per_microbatch=-1",
        "trainer.gradient_checkpointing=false",
    ]
    monkeypatch.setattr(sys, "argv", ["run_dapo", "--data-dir", "/data", "--run-dir", "/run", "--dry-run", *overrides])
    main()
    command = shlex.split(capsys.readouterr().out)
    assert command[-len(overrides) :] == overrides


def test_short_run_rejects_production_warmup_before_launch(monkeypatch, capsys):
    import sys

    import pytest

    from examples.train.isoexec.run_dapo import main

    monkeypatch.setattr(
        sys, "argv", ["run_dapo", "--data-dir", "/data", "--run-dir", "/run", "trainer.max_training_steps=2"]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "num_warmup_steps=0" in capsys.readouterr().err


def test_dapo_recipe_resolves_production_schedule_and_optimizer():
    from examples.train.algorithms.dapo.main_dapo import DAPOConfig

    cfg = DAPOConfig.from_cli_overrides(dapo_overrides(Path("/data"), Path("/run")))
    assert cfg.trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32 is True
    assert cfg.trainer.max_training_steps == 50
    assert cfg.trainer.policy.optimizer_config.num_warmup_steps == 40
    assert cfg.trainer.policy.optimizer_config.weight_decay == 0.1
    optimizer = cfg.trainer.policy.megatron_config.optimizer_config_kwargs
    assert optimizer["overlap_cpu_optimizer_d2h_h2d"] is True
    assert optimizer["pin_cpu_grads"] is True
    assert optimizer["pin_cpu_params"] is True
    assert cfg.trainer.algorithm.off_policy_correction.tis_ratio_type == "token"
    assert cfg.trainer.rollout_logprob_comparison == "action"
    assert cfg.generator.inference_engine.logprob_output == "action"
    assert cfg.generator.inference_engine.distributed_executor_backend == "mp"
    assert cfg.generator.sampling_params.min_tokens == 1
    assert cfg.isoexec_model["revision"] == "0f0813072d2358973511097385626f21fcb6d422"
