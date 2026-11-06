# IsoExec recipes

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

## Qwen3.5-35B-A3B DAPO

The production recipe uses eight H100 80GB GPUs, trainer TP=4/EP=8 with
sequence parallelism, and a colocated TP=8 inference engine. The built-in
`Qwen/Qwen3.5-35B-A3B-Base` checkpoint is pinned to
`0f0813072d2358973511097385626f21fcb6d422`. A supplied `--checkpoint` is used
as-is; local directories must contain that snapshot for the same starting weights.

Run it from the SkyRL checkout that IsoExec's `make setup` creates, with the
environment exports from [Setup](#setup).

**Qualification.** [Run cdz5z7l5](https://wandb.ai/zander-jiang/qwen3.5-35b-dapo/runs/cdz5z7l5)
used an earlier stack: IsoExec `repro/production-35b-environment`
(`environments/production-35b.toml`) with SkyRL
`repro/production-dapo-integration-20261104`. For an exact replay, use that pair.
This branch carries the same recipe (the resolved overrides are identical) on the
current IsoExec runtime, which has not yet been qualified by an eight-GPU run.

Prepare the two cleaned parquet files described in
[production-inputs.json](production-inputs.json), authenticate W&B through your
environment, then run from the prepared SkyRL integration checkout:

```bash
python -m examples.train.isoexec.run_dapo \
  --data-dir /mnt/local_storage/data/dapo \
  --run-dir /mnt/local_storage/runs/isoexec-qwen35-dapo
```

Use `--dry-run` to inspect
the command; trailing `key=value` arguments override the defaults. The recipe
selects these production settings:

| Setting | Value |
|---|---|
| Schedule | 50 steps, 40 warmup steps, learning rate `1e-6`, weight decay `0.1` |
| Rollouts | 128 prompts × 16 samples = 2,048 sequences |
| Tokens | prompt 2,048; response 8,192; packed microbatch budget 20,480 |
| Updates | four optimizer minibatches per rollout; full decoder-layer checkpointing |
| Gradients and optimizer | FP32 reduction; CPU offload, pinned parameters/gradients, transfer overlap |
| Scores | exact sampled-action comparison, `logprobs=0`, token-level TIS |
| Inference | multiprocessing, 512 maximum sequences, decode CUDA Graphs |
| Evaluation | interval 1,000; 32 samples, top-p `0.7`; none scheduled within 50 steps |
| Persistence | fresh start; training-state save/resume and HF export disabled |

The CISPO bounds (`0`, `5`) are retained from the recorded configuration; the
selected policy loss is `dual_clip`, so those bounds do not select CISPO loss.
The generic `run.py` retains its separate full-vocabulary comparison recipe.
Native transfer verification defaults off; enable
`ISOEXEC_VERIFY_NATIVE_TRANSFERS=1` only when investigating synchronization.

### Input provenance

The input manifest records immutable raw-download URLs, raw and cleaned SHA256
hashes, and row counts. Hugging Face LFS metadata at those revisions matches the
raw files used by the run: DAPO has 1,791,700 raw rows and 17,391 cleaned rows;
AIME-2024 has 960 raw rows and 30 cleaned rows. Place the raw files under their
manifest filenames and run the existing preprocessor:

```bash
python -m examples.train.algorithms.dapo.data_preprocess_dapo_aime \
  --data-dir /mnt/local_storage/data/dapo
```

It deduplicates `(data_source, prompt, ability, reward_model)` and removes prompts
with multiple reward models. Its Polars `unique` call does not preserve row order
by default. Regenerated parquet bytes and training order may therefore differ;
**exact replay requires the archived cleaned files matching the manifest hashes**.
The repository does not distribute those cleaned artifacts. The generic
`prepare_dapo_data.sh` uses mutable raw URLs; use the manifest URLs for this run.

### Production qualification

[Run cdz5z7l5](https://wandb.ai/zander-jiang/qwen3.5-35b-dapo/runs/cdz5z7l5)
completed all 50 training steps and 50 exact prescore checks on 2026-11-04,
with 51 weight-sync phases including initialization and process exit code zero.
The detailed per-rank finite-gradient/optimizer audit covers the first two cycles;
no OOM or fatal errors were reported. This followed an earlier 25-step run that
was intentionally stopped for performance work.

The recipe defaults reproduce the recorded launch settings; run names and output
paths are user-specific. The raw model and data pins identify the inputs, but
backward bitwise determinism remains unqualified: repeated uncheckpointed
controls also varied. The recorded success does not qualify different token
budgets, parallel layouts, checkpoint spans, a newly installed environment, or
the current IsoExec runtime.

### MoE inference performance

The DAPO launcher enables `ISOEXEC_MOE_AFTER_FC2_OWNER=1` and
`ISOEXEC_MOE_AFTER_FC2_GATHER=1`. These restore fused expert gathering and combine
paths while preserving the routing-weight multiplication after FC2 and its
rounding order. Other IsoExec entrypoints leave these optimizations opt-in.
Set either variable to `0` before launching to disable that path.

Matched eight-H100, TP=8 tests with the same native model, prompts, and sampling
settings measured 1.28× decode throughput and 1.22× mixed-prefill throughput.
The decode fixture used 512 requests with 512 output tokens each; the mixed
fixture used 1,024 requests with 64–256 output tokens. Three measured rounds
followed warmup for each fixture. Generated tokens and sampled-token logprobs
matched exactly in every round, including after sleep, weight update, wake-up,
and weight restoration. These measurements cover bounded generation workloads;
production step time also includes longer rollouts, scoring, and training.

### Activation checkpoint settings

`trainer.gradient_checkpointing=true` enables native `full` / `uniform`
recomputation. The default span is one complete decoder layer, including its
norms, attention or GDN, MoE/MLP, and residual paths. Embeddings, the final norm,
and the LM head remain outside these regions. The 35B DAPO recipe enables this
flag; the smaller-model `run.py` recipe leaves it disabled.

Set `recompute_num_layers` to a positive integer to group consecutive layers.
For example, append these overrides to
`python -m examples.train.isoexec.run_dapo` to checkpoint two layers at a
time while retaining its sequence-parallel setting:

```bash
trainer.gradient_checkpointing=true \
'trainer.policy.megatron_config.transformer_config_kwargs={"sequence_parallel":true,"recompute_num_layers":2}'
```

A final partial group checkpoints the remaining layers on that pipeline stage.
The eight-GPU qualification used span `1`; larger spans have CPU equivalence
coverage but need workload-specific memory and performance checks. SkyRL
selects `full`/`uniform` when its checkpointing flag is enabled. Direct native
runtime calls reject `block` or `selective` modes and boolean layer spans; use
an integer span such as `1`.

Recomputation uses Megatron's RNG-aware checkpoint primitive. Standard SkyRL
setup seeds its CUDA RNG tracker; direct runtime callers must initialize that
tracker before training. This preserves RNG replay, without making inherently
nondeterministic kernels bitwise deterministic.

Explicit MoE auxiliary outputs retain their gradients; native SkyRL auxiliary
loss currently requires PP=1. The DAPO recipe uses auxiliary coefficient zero.
Keep the standard SkyRL optimizer, offload/onload, and weight-sync lifecycle:
checkpoint regions share the original registered parameters. Activation
checkpointing does not enable training-state save/resume or HF export.

For a two-step training smoke test, override the 40-step production warmup as well
as the training limit. Allow 4,096 response tokens so math solutions have room to
finish, and use one sequence per microbatch to limit activation memory without
activation checkpointing:

```bash
python -m examples.train.isoexec.run_dapo \
  --data-dir /mnt/local_storage/data/dapo \
  --run-dir /mnt/local_storage/runs/isoexec-qwen35-smoke \
  trainer.max_training_steps=2 trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.rollout_logprob_comparison=action generator.inference_engine.logprob_output=action \
  generator.sampling_params.max_generate_length=4096 \
  generator.eval_sampling_params.max_generate_length=4096 trainer.algorithm.overlong_buffer_len=1024 \
  trainer.max_tokens_per_microbatch=-1 trainer.gradient_checkpointing=false
```

This keeps the 2,048-sequence rollout batch and tests sampled-token comparison;
it does not qualify the full-vocabulary comparison path. The production recipe
retains its 40-step warmup. The 2,048-token prompt limit plus 4,096 response
tokens fits the existing 10,240-token engine context. With token packing disabled,
the existing forward and training microbatch sizes of one sequence apply.

Keep overlong filtering and the comparison audit enabled, and allow enough
response tokens for untruncated, trainable answers. A longer cap increases the
chance of completed answers; it does not guarantee them. The 4,096-token smoke
run has not completed end-to-end qualification.
