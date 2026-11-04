# IsoExec Model V2: GSM8K

Megatron trains the authored IsoExec model; vLLM serves its weights. Both use the
same HF checkpoint `config.json`. Model dimensions and layer types come from
that file; the example selects only the architecture package.

## Setup

Use Linux, Python 3.12, CUDA 13 and an NVIDIA GPU. Start with Qwen3.5-0.8B on
one H100. The larger models need sufficient GPU and host memory; CPU optimizer
offload is enabled. This branch is based on SkyRL's IsoExec integration branch.

```bash
git clone --branch codex/isoexec-v2-examples https://github.com/Future-Outlier/SkyRL.git
git clone --branch codex/skyrl-root-config https://github.com/Future-Outlier/IsoExec.git
git clone https://github.com/fla-org/flash-linear-attention.git
git -C flash-linear-attention checkout ebf3a0cff2be3e6f2b2f99820b8fe4e28855ced0
cd SkyRL
export ISOEXEC_ROOT="$(cd ../IsoExec && pwd)"
export ISOEXEC_ENV="$HOME/.cache/skyrl-isoexec-v2"

uv run --isolated --no-project "$ISOEXEC_ROOT/tools/bootstrap_env.py" \
  --root "$ISOEXEC_ENV" --skyrl-source "$PWD" --fla-source "$(cd ../flash-linear-attention && pwd)"
uv pip install --python "$ISOEXEC_ENV/envs/core/bin/python" --reinstall --no-deps \
  'nvidia-cutlass-dsl[cu13]==4.5.2' \
  'nvidia-cutlass-dsl-libs-base==4.5.2' 'nvidia-cutlass-dsl-libs-cu13==4.5.2'
source "$ISOEXEC_ENV/env.sh"

uv run --isolated --no-project "$ISOEXEC_ENV/envs/core/bin/python" \
  examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
```

The runtime requires IsoExec's native OP dependencies, including FlashAttention 3
and CUTLASS DSL 4.5.2. The explicit CUTLASS install above is required because the
current bootstrap defaults to 4.7.1, which is not admitted for these models'
RMSNorm binding. This example uses the current project vLLM 0.30.0 and
Transformers 5.16.1. Do not install the old vLLM 0.28.0 qualification SDK.

## Shared IsoExec settings

```yaml
# IsoExec settings shared by Megatron training and vLLM inference.
enable_isoexec: true
isoexec_model:
  source:
    api_version: 2
    package: isoexec.models.v2.qwen3_5
```

## Run

```bash
uv run --isolated --no-project "$ISOEXEC_ENV/envs/core/bin/python" \
  examples/train/isoexec_v2/run.py --model qwen3_5_0_8b
```

The default is one optimizer update with full FP32 vocabulary logprob comparison
and decode CUDA Graphs. GSM8K limits are prompt 512, generation 1024 and model
length 2048. Model V2 is selected with `isoexec_model.source.api_version=2`;
it does not select vLLM Model Runner V2.
The launcher allows 1200 seconds for server startup so first-time native kernel
builds can finish before SkyRL's health check expires.

| `--model` | Checkpoint | Shared model package |
|---|---|---|
| `qwen3_5_0_8b` | `Qwen/Qwen3.5-0.8B` | `isoexec.models.v2.qwen3_5` |
| `qwen3_5_9b` | `Qwen/Qwen3.5-9B` | `isoexec.models.v2.qwen3_5` |
| `qwen1_5_moe_a2_7b` | `Qwen/Qwen1.5-MoE-A2.7B` | `isoexec.models.v2.qwen2_moe` |

The GSM8K Hub IDs use pinned revisions. The driver reads that snapshot's HF
`config.json` and safetensors and sends the resolved configuration to workers.
Qwen3.5 executes the text decoder; vision and MTP are excluded.

For a local/private checkpoint with the same architecture:

```bash
uv run --isolated --no-project "$ISOEXEC_ENV/envs/core/bin/python" \
  examples/train/isoexec_v2/run.py --model qwen3_5_0_8b \
  --checkpoint /models/private-qwen --data-dir /data/gsm8k --run-dir /data/run
```

That directory must contain its HF `config.json`, safetensors and tokenizer
files. Python `build_config` hooks and `model_factory` selectors are not used.
Append ordinary SkyRL `key=value` overrides for native settings. `--dry-run`
prints the exact command without starting Ray or allocating a GPU.

Native training checkpoint save/resume and HF export remain disabled by the
current IsoExec V2 integration. Console logging is the default; select W&B with
`trainer.logger=wandb` and your own credentials.

## Qwen3.5-35B-A3B DAPO

Requires the companion IsoExec change `c99f000` (or a successor containing
`isoexec.models.v2.qwen3_5_moe`) and eight H100 GPUs. The 35B Hub ID tracks its
default revision; use a local `--checkpoint` snapshot for reproducible runs.
Prepare cleaned DAPO/AIME parquet files using the DAPO data-preparation example,
and authenticate W&B through the environment before launching:

```bash
uv run --isolated --extra megatron -m examples.train.isoexec_v2.run_dapo \
  --data-dir /mnt/local_storage/data/dapo \
  --run-dir /mnt/local_storage/runs/isoexec-qwen35-dapo
```

Use `--dry-run` to inspect the command or `--checkpoint` for a local checkpoint.
The rollout batch is 128 prompts × 16 samples = 2,048 sequences. Trainer TP=4,
EP=8 and sequence parallelism share eight GPUs with a TP=8 inference engine.
Activation checkpointing is enabled for the 35B recipe; checkpoint export remains
disabled. This requires IsoExec's native whole-layer activation checkpointing
support. Native transfer verification follows IsoExec's default (off); opt in with
`ISOEXEC_VERIFY_NATIVE_TRANSFERS=1` when investigating synchronization.

This uses the current upstream GDN provider (CPR). On 2026-11-03, the Qwen3.5
35B-A3B recipe completed two production DAPO cycles on eight H100 GPUs with
trainer TP=4, EP=8, sequence parallelism, FP32 gradient buffers, a 20,480-token
microbatch budget, and full/uniform checkpoint span `1`. Each cycle completed
its 2,048-response rollout, exact action-logprob prescore check, four successful
optimizer updates per rank with finite gradients, offload/onload and weight sync,
then began the next rollout. The run subsequently reached 25 complete training
steps and 26 exact prescore checks without reported OOM or fatal errors before
being intentionally stopped for performance work. The detailed per-rank optimizer
audit covers the first two cycles; the planned 50 steps were not completed.

The trainer's offloaded GPU-memory floor was stable within 2 MiB per process
between the second and third rollouts after its initial warmup increase. This
is bounded evidence for this recipe, not a guarantee for other models, spans,
parallel layouts, token budgets, or long-run memory behavior. Backward bitwise
determinism remains unqualified; repeated uncheckpointed controls also varied.
Training-state save/resume and HF export remained disabled in this run.

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
`uv run --isolated --extra megatron -m examples.train.isoexec_v2.run_dapo` to checkpoint two layers at a
time while retaining its sequence-parallel setting:

```bash
trainer.gradient_checkpointing=true \
'trainer.policy.megatron_config.transformer_config_kwargs={"sequence_parallel":true,"recompute_num_layers":2}'
```

A final partial group checkpoints the remaining layers on that pipeline stage.
The eight-GPU qualification used span `1`; larger spans have CPU equivalence
coverage but need workload-specific memory and performance checks. Explicit
`block` or `selective` modes are rejected. When checkpointing is enabled, the
native runtime rejects boolean layer spans; use an integer such as `1`.

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
uv run --isolated --extra megatron -m examples.train.isoexec_v2.run_dapo \
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

A 256-token cap produced only truncated, loss-masked responses across a complete
2,048-sequence batch. Native weight sync, generation, and trainer forward scoring
completed, but the comparison audit correctly refused a batch with no trainable
response tokens. Keep overlong filtering and this audit enabled. A longer cap
increases the chance of completed answers; it does not guarantee them. The
4,096-token smoke run has not yet completed end-to-end qualification.
