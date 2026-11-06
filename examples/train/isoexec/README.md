# IsoExec: GSM8K

Megatron trains the authored IsoExec model; vLLM serves its weights. Both use the
same HF checkpoint `config.json`. Model dimensions and layer types come from
that file; the example selects only the architecture package.

## Setup

Use Linux x86-64, CUDA 13 and NVIDIA H100s. Start with Qwen3.5-0.8B on one GPU.
The larger models need sufficient GPU and host memory; CPU optimizer offload is
enabled.

IsoExec's setup installs the pinned runtime (Python, vLLM, Megatron, FlashAttention 3,
CUTLASS headers and kernel overrides) and checks out this SkyRL branch at the
revision IsoExec pins:

```bash
git clone https://github.com/FirstLight-org/IsoExec.git && cd IsoExec
export ISOEXEC_ROOT=/mnt/local_storage/isoexec   # a new, empty directory
make setup ISOEXEC_ROOT="$ISOEXEC_ROOT"
make run-tiny ISOEXEC_ROOT="$ISOEXEC_ROOT"     # Qwen3.5-0.8B, one GRPO step, one GPU
```

To run the recipes below, use the SkyRL checkout setup created and propagate the
runtime environment to Ray workers:

```bash
cd "$ISOEXEC_ROOT/skyrl"
export PATH="$ISOEXEC_ROOT/venv/bin:$PATH" UV_PROJECT_ENVIRONMENT="$ISOEXEC_ROOT/venv"
export HF_HOME="$ISOEXEC_ROOT/cache/hf" CUDA_DEVICE_MAX_CONNECTIONS=1
export SKYRL_PYTHONPATH_EXPORT=1 SKYRL_LD_LIBRARY_PATH_EXPORT=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
python examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
```

Do not run `uv sync` in this checkout afterwards: it would replace the runtime's
deliberate overrides.

## Shared IsoExec settings

```yaml
# IsoExec settings shared by Megatron training and vLLM inference.
enable_isoexec: true
isoexec_model:
  source:
    package: isoexec.models.qwen3_5
```

## Run

```bash
python examples/train/isoexec/run.py --model qwen3_5_0_8b
```

The default is one optimizer update with full FP32 vocabulary logprob comparison
and decode CUDA Graphs. GSM8K limits are prompt 512, generation 1024 and model
length 2048. The model package is selected with `isoexec_model.source.package`.
The launcher allows 1200 seconds for server startup so first-time native kernel
builds can finish before SkyRL's health check expires.

| `--model` | Checkpoint | Shared model package |
|---|---|---|
| `qwen3_5_0_8b` | `Qwen/Qwen3.5-0.8B` | `isoexec.models.qwen3_5` |
| `qwen3_5_9b` | `Qwen/Qwen3.5-9B` | `isoexec.models.qwen3_5` |
| `qwen1_5_moe_a2_7b` | `Qwen/Qwen1.5-MoE-A2.7B` | `isoexec.models.qwen2_moe` |

Built-in Hub IDs use pinned revisions. The driver reads that snapshot's HF
`config.json` and safetensors and sends the resolved configuration to workers.
Qwen3.5 executes the text decoder; vision and MTP are excluded.

For a local/private checkpoint with the same architecture:

```bash
python examples/train/isoexec/run.py --model qwen3_5_0_8b \
  --checkpoint /models/private-qwen --data-dir /data/gsm8k --run-dir /data/run
```

That directory must contain its HF `config.json`, safetensors and tokenizer
files. Python `build_config` hooks and `model_factory` selectors are not used.
Append ordinary SkyRL `key=value` overrides for native settings. `--dry-run`
prints the exact command without starting Ray or allocating a GPU.

Native training checkpoint save/resume and HF export remain disabled by the
current IsoExec integration. Console logging is the default; select W&B with
`trainer.logger=wandb` and your own credentials.
