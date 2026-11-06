"""The shared example resolves real HF configs through the IsoExec driver boundary."""

import json
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from skyrl.train.config import SkyRLTrainConfig

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/train/isoexec/run.py"
example = runpy.run_path(str(EXAMPLE))
MODELS, training_overrides = example["MODELS"], example["training_overrides"]


def _qwen35_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 16,
            "vocab_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 32,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 0, 0],
                "mrope_interleaved": True,
            },
        },
    }


def test_checkpoint_config_selects_the_shared_architecture(tmp_path):
    pytest.importorskip("isoexec.models.qwen3_5.model")
    from isoexec.integrations.skyrl.full_distribution_config import (
        validate_full_distribution_config,
    )
    from isoexec.integrations.skyrl.model_folder import (
        prepare_checkpoints,
        source_config,
    )

    name = "qwen3_5_0_8b"
    raw = _qwen35_config()
    checkpoint = tmp_path / "private checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(raw))
    # Artifact resolution checks the header; this test constructs only on meta.
    save_file({"probe": torch.ones(1)}, str(checkpoint / "model.safetensors"))
    cfg = SkyRLTrainConfig.from_cli_overrides(
        training_overrides(name, tmp_path / "data", tmp_path / "run", checkpoint=str(checkpoint))
    )
    validate_full_distribution_config(cfg)
    prepare_checkpoints(cfg)
    selected = source_config(cfg.isoexec_model, cfg.trainer, str(checkpoint))
    assert selected is not None
    source, effective = selected
    assert effective == raw
    assert source.package == MODELS[name][1]
    model = source.build_meta(effective)
    architectures = raw["architectures"]
    assert isinstance(architectures, list)
    assert type(model).__name__ in architectures
    assert all(parameter.is_meta for parameter in model.parameters())
    engine = cfg.generator.inference_engine
    assert not engine.enforce_eager
    assert engine.engine_init_kwargs["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    assert engine.engine_init_kwargs["max_model_len"] == 2048
    assert cfg.trainer.max_prompt_length == 512
    assert cfg.generator.sampling_params.max_generate_length == 1024


def test_presets_select_architecture_packages_without_bundled_checkpoint_configs():
    assert MODELS["qwen3_5_0_8b"][1] == MODELS["qwen3_5_9b"][1] == "isoexec.models.qwen3_5"
    assert MODELS["qwen1_5_moe_a2_7b"][1] == "isoexec.models.qwen2_moe"


def test_dry_run_keeps_paths_and_native_overrides_without_starting_training(tmp_path):
    data = tmp_path / "data with spaces"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--dry-run", "--data-dir", str(data), "trainer.max_training_steps=2"],
        check=True,
        capture_output=True,
        text=True,
    )
    command = shlex.split(result.stdout)
    assert command[:3] == [
        "env",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV=0",
        "SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1200",
    ]
    assert command[3:6] == [sys.executable, "-m", "skyrl.train.entrypoints.main_base"]
    cfg = SkyRLTrainConfig.from_cli_overrides(command[6:])
    assert cfg.data.train_data == [str(data / "train.parquet")]
    assert cfg.trainer.max_training_steps == 2
    assert cfg.isoexec_model is not None
    assert cfg.isoexec_model["revision"] == MODELS["qwen3_5_0_8b"][2]
    assert cfg.isoexec_model["source"]["package"] == "isoexec.models.qwen3_5"
    assert not any("model_factory" in argument for argument in command)
