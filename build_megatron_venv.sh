#!/usr/bin/env bash
set -x
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export UV_PROJECT_ENVIRONMENT=/home/ray/venvs/skyrl
cd /home/ray/default/SkyRL
echo "BUILD_START $(date -u)"
uv sync --extra megatron
code=$?
echo "BUILD_DONE exit=$code $(date -u)"
