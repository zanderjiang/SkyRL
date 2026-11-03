"""The production recipe keeps native placement and rollout sizing explicit."""

import json
from pathlib import Path

from examples.train.isoexec_v2.run_dapo import dapo_overrides


def test_dapo_recipe_has_2048_rollouts_and_no_checkpointing():
    values = {
        key: json.loads(value)
        for item in dapo_overrides(Path("/data"), Path("/run"))
        for key, value in [item.split("=", 1)]
    }
    assert values["enable_isoexec"] is True
    assert values["isoexec_model"]["source"]["package"] == "isoexec.models.v2.qwen3_5_moe"
    assert values["trainer.train_batch_size"] * values["generator.n_samples_per_prompt"] == 2048
    assert values["trainer.gradient_checkpointing"] is False
    assert values["trainer.ckpt_interval"] == values["trainer.hf_save_interval"] == -1
    assert values["trainer.policy.megatron_config.tensor_model_parallel_size"] == 4
    assert values["trainer.policy.megatron_config.expert_model_parallel_size"] == 8
    assert values["generator.inference_engine.tensor_parallel_size"] == 8
    assert values["trainer.policy.megatron_config.transformer_config_kwargs"] == {"sequence_parallel": True}


def test_smoke_overrides_reach_dry_run(monkeypatch, capsys):
    import shlex
    import sys

    from examples.train.isoexec_v2.run_dapo import main

    overrides = [
        "trainer.max_training_steps=2",
        "trainer.policy.optimizer_config.num_warmup_steps=0",
        "trainer.rollout_logprob_comparison=action",
        "generator.inference_engine.logprob_output=action",
        "generator.sampling_params.max_generate_length=4096",
        "trainer.algorithm.overlong_buffer_len=1024",
        "generator.eval_sampling_params.max_generate_length=4096",
        "trainer.max_tokens_per_microbatch=-1",
    ]
    monkeypatch.setattr(sys, "argv", ["run_dapo", "--data-dir", "/data", "--run-dir", "/run", "--dry-run", *overrides])
    main()
    command = shlex.split(capsys.readouterr().out)
    assert command[-len(overrides) :] == overrides


def test_short_run_rejects_production_warmup_before_launch(monkeypatch, capsys):
    import sys

    import pytest

    from examples.train.isoexec_v2.run_dapo import main

    monkeypatch.setattr(
        sys, "argv", ["run_dapo", "--data-dir", "/data", "--run-dir", "/run", "trainer.max_training_steps=2"]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "num_warmup_steps=0" in capsys.readouterr().err


def test_dapo_recipe_inherits_fp32_gradient_accumulation():
    from examples.train.algorithms.dapo.main_dapo import DAPOConfig

    cfg = DAPOConfig.from_cli_overrides(dapo_overrides(Path("/data"), Path("/run")))
    assert cfg.trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32 is True
