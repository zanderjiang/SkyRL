set -x

# ==========================================================================================
# GATE 3.3 -- live zero-KL GRPO on Qwen/Qwen3.5-0.8B (GatedDeltaNet hybrid) + GSM8K, DP8.
# ==========================================================================================
# This is the last GDN-specific gate. Everything it depends on is already proven offline:
#   * GDN decode == prefill, bitwise, on the native vLLM engine   (65536/65536, max 0.0)
#   * the same through the production GPTModel-in-vLLM engine     (256/256, max 0.0, coherent)
#   * the trainer builds the hybrid, loads it, fwd+bwd            (predicts ' Rome')
#   * dense (MiMo-7B) and MoE (OLMoE) engine parity unregressed   (256/256, max 0.0)
# See examples/zerokl/nightly/GDN_ZEROKL_REPORT.md for the commands and logs.
#
# THE GATE: wandb `policy/rollout_train_logprobs_abs_diff_mean` <= 1e-6 at EVERY step, including
# steps 2-5 (post weight update), and `policy_kl` == 0.0. A clean step 1 with a dirty step 2 is the
# sleep/wake weight-clobber class of bug, not a GDN bug -- set SKYRL_ZEROKL_DEBUG=1 and compare the
# [ZEROKL-REAPPLY] == [SENDER] == [ZEROKL-ENGFWD] checksums.
#
# Matched TP>1 (ZEROKL_TP=2 here) is the staging ground for Qwen3.5-35B-A3B, which needs matched
# TP=8: 70 GB of bf16 weights cannot be held twice (trainer + engine) on one 80 GB GPU at TP=1.
# The wrapper builds a TP-sharded GPTModel over vLLM's worker group (gptmodel_vllm.py), the native
# sync ships per-rank shards via GPU-keyed CUDA-IPC, and the trainer's logprob extraction gathers
# the full vocab row to bitwise-match the engine's gathered-logits formula (model_utils.py).
#
# Rollout is ~5.8x slower than stock vLLM at 16 concurrent sequences: chunk-consistent decode
# re-runs the training chunk kernel over the open chunk, and today does it in a per-slot python loop.
# Expect slow steps. See the report's cost section.
#
# Prepare data first:
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# Launch:
#   WANDB_API_KEY=<key> bash examples/zerokl/run_megatron_qwen3.5_0.8b_gsm8k_zerokl_nightly.sh \
#       > /mnt/local_storage/logs/zerokl_gsm8k_qwen35_0.8b.log 2>&1

MODEL_NAME="${QWEN35_MODEL_PATH:-Qwen/Qwen3.5-0.8B}"
DATA_DIR="$HOME/data/gsm8k"
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/validation.parquet"

NUM_NODES=1
NUM_GPUS_PER_NODE=8
# TP is matched on both ends (zero-KL requirement). ZEROKL_TP lets us exercise TP>1 on the small
# model before committing 8 GPUs to a 35B run.
MEGATRON_TP=${ZEROKL_TP:-1}
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=$MEGATRON_TP
NUM_INFERENCE_ENGINES=$((NUM_GPUS_PER_NODE / MEGATRON_TP))
MEGATRON_PP=1
MEGATRON_CP=1

LOGGER="${ZEROKL_LOGGER:-wandb}"
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
TRAIN_BATCH_SIZE=64
MINI_BATCH_SIZE=64
N_SAMPLES_PER_PROMPT=4
LR=1e-6
ENFORCE_EAGER=true
REMOVE_MICROBATCH_PADDING=false
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Qwen3.5 registers as a VL architecture; language_model_only routes the bridge to the plain GPTModel
# (+ GDN) instead of the VL model.
LANGUAGE_MODEL_ONLY=True

# ===== zero-KL switches =====
export SKYRL_ZERO_KL=1
export SKYRL_ZEROKL_LOCAL_SPEC=1
export SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=1
export SKYRL_ZEROKL_GDN=1                  # fla shim + hybrid no-TE spec + chunk-consistent decode
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VARLEN_FORCE_NUM_SPLITS_1=1
# Chunk-consistent GDN decode redefines ssm_state[slot] as the state at the last CHUNK BOUNDARY.
# Prefix caching, chunked prefill and CUDA graphs all reinterpret it; the engine raises rather than
# degrade quietly. They are bitwise-safe for the softmax layers (worth 4.6x rollout) -- re-enabling
# them for GDN is a follow-up, not a bug.
export SKYRL_ZEROKL_ENABLE_PREFIX_CACHE=0
export SKYRL_ZEROKL_ENABLE_CHUNKED_PREFILL=0
export SKYRL_ZEROKL_ENABLE_CUDAGRAPH=0
export SKYRL_ZEROKL_NO_CHUNKED_PREFILL=1
export SKYRL_ZEROKL_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# vLLM's Qwen3.5 module imports the VL chain unconditionally; torchvision is absent from the zerokl
# venv. The stub only has to satisfy the import.
export PYTHONPATH="$(cd "$(dirname "$0")/nightly/_torchvision_stub" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/mnt/local_storage/hf}"
DISTRIBUTED_EXECUTOR_BACKEND="mp"

uv run --isolated --extra zerokl -m skyrl.train.entrypoints.main_base \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.off_policy_correction.tis_ratio_type=null \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=${ZEROKL_MB:-1} \
  trainer.micro_train_batch_size_per_gpu=${ZEROKL_MB:-1} \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=5 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.42 \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen35_0.8b_gsm8k_zerokl" \
  trainer.run_name="zerokl_gdn_qwen35_0.8b_dp${NUM_GPUS_PER_NODE}" \
  trainer.export_path="/mnt/local_storage/exports/zerokl_gdn_qwen35_0.8b" \
  trainer.resume_mode=none \
  trainer.ckpt_path="/mnt/local_storage/ckpts/zerokl_gdn_qwen35_0.8b" \
  "$@"
