#!/usr/bin/env bash
# Wrapper to launch the Qwen3.5-9B MTP spec-decode DAPO run at commit 8e6b355.
set -x
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Point uv/ray at the base venv built in /home/ray
export UV_PROJECT_ENVIRONMENT=/home/ray/venvs/skyrl
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
# Faster HF downloads for the 9B checkpoint
export HF_HUB_ENABLE_HF_TRANSFER=1
# wandb: WANDB_API_KEY is inherited from the launching environment (not printed here)
: "${WANDB_API_KEY:?set WANDB_API_KEY before running}"

cd /home/ray/default/SkyRL
exec bash examples/train/megatron/run_megatron_dapo_qwen3.5_9b_specdecode.sh "$@"
