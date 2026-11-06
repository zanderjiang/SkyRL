# IsoExec Qwen3.5-0.8B training configuration

Activate a prepared Linux/H100 environment with this SkyRL branch and IsoExec
installed, including its Megatron/vLLM kernel dependencies. Set
`ISOEXEC_FLA_SOURCE` to the required FLA checkout. The command uses the active
environment without reinstalling CUDA packages.

From the SkyRL repository root, on four GPUs:

```bash
bash test_examples/isoexec/run_isoexec_qwen3.5_0.8b.sh
```

For PP2 on eight GPUs:

```bash
PP=2 bash test_examples/isoexec/run_isoexec_qwen3.5_0.8b.sh
```

The shell contains the complete configuration: trainer TP2/SP/dense DP2,
two independent inference engines each using TP2 and the same PP, BF16,
text-only, MTP/speculation off, and LoRA rank 0. The model package is
`isoexec.models.qwen3_5`; the official checkpoint revision is pinned.
Both full-logprob settings are enabled so policy prescore compares complete FP32
vocabulary rows before the optimizer update.

This is a normal GSM8K GRPO training launch through
`skyrl.train.entrypoints.main_base`. Data is prepared automatically if missing.
It performs one training iteration by default. It uses the parallelism topology
from the previous Modal qualification, but the GSM8K data/context settings are
a training example, not a reproduction of the original synthetic W0/W1 verifier.

`DATA_DIR` selects the GSM8K folder (default `~/data/gsm8k`).
`RUN_DIR` selects a fresh output folder (default under `/tmp`).
`ISOEXEC_ROOT` selects the checkout (default sibling `../IsoExec`);
`ISOEXEC_PYTHON` selects a prepared Python executable.
Ray starts locally unless `RAY_ADDRESS` is already set.
Append normal SkyRL overrides, for example `trainer.max_training_steps=10`.

To print the command without launching training:

```bash
DRY_RUN=1 bash test_examples/isoexec/run_isoexec_qwen3.5_0.8b.sh
```

This example contains no Modal submission code, frozen source bundle, or
qualification artifacts. It runs the installed/current IsoExec code.
