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
