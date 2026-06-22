set -x

# Fully async (disaggregated) DAPO training+generation for Qwen3.5-4B (dense) with Megatron.
# Trainer and rollout engines run on SEPARATE GPUs concurrently (colocate_all=false):
#   - Policy (trainer): 2 GPUs, Megatron TP=2 (PP=1, CP=1 => DP=1).
#   - Generation: 4 vLLM inference engines, TP=1 (1 GPU each).
#   - Total = 2 + 4 = 6 GPUs, fits on 1 node of 8xH100s (80GB each) with 2 GPUs to spare.
#
# Because training and generation are NOT colocated, the policy GPUs never host the
# inference engine, so the full 80GB/GPU is available for the optimizer -- TP=2 fits 4B.
#
# NOTE: verify the exact HF repo id for the 4B model before running
#   (e.g. `hf download Qwen/Qwen3.5-4B` / check https://huggingface.co/Qwen).
#
# Prepare data onto the fast local disk first:
#   DATA_DIR=/mnt/local_storage/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# Then launch:
#   bash examples/train/megatron/run_megatron_dapo_qwen3.5_4b_async.sh

MODEL_NAME="Qwen/Qwen3.5-4B"
# Use the fast, non-persistent local disk for data (not the ~/default quota).
DATA_DIR="/mnt/local_storage/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"

NUM_NODES=1
# Disaggregated placement: dedicated trainer GPUs + dedicated inference GPUs.
NUM_POLICY_GPUS=2
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1
LOGGER="wandb"  # change to "console" to print to stdout

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
# use token mean loss reduction
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl in main_dapo_fully_async.py
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# other DAPO parameters
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))

# repro run parameters
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=16
ENFORCE_EAGER=true # cuda graphs can cause some instability
LR=1e-6

# Fully async specific configuration knobs:
#   max_staleness_steps: how many policy steps the rollouts may lag behind (1 = one-step off-policy).
#   num_parallel_generation_workers: number of concurrent rollout requests in flight.
: "${MAX_STALENESS_STEPS:=1}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=64}"

# Off-policy correction (geometric sequence masking) -- async rollouts are slightly stale.
SEQUENCE_MASK_METRIC=geometric
GEO_MASK_HIGH=1.01
GEO_MASK_LOW=0.99

# megatron config -- Qwen3.5-4B is a dense model, so no expert parallelism.
# TP=2 (PP=1, CP=1) => DP=1. TP>1 auto-enables sequence parallelism, which shards
# activations/vocab-logits across the TP group so 8K-token responses fit at micro batch 1.
# TP comm stays within the single-node NVLink domain.
MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=null

# TIS parameters
TIS_IMP_RATIO_CAP=2.0
TIS_TYPE=token

# Qwen3.5 flags
REMOVE_MICROBATCH_PADDING=false # sample packing is not yet supported for GDN layers in megatron - see: https://github.com/NVIDIA/Megatron-LM/pull/2644
ENGINE_INIT_KWARGS='{"gdn_prefill_backend": "triton"}' # see https://github.com/vllm-project/vllm/issues/36921#issuecomment-4109702738
DISTRIBUTED_EXECUTOR_BACKEND="mp"
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

RUN_NAME="async_dapo_qwen3_5_4b_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_${NUM_POLICY_GPUS}train${NUM_INFERENCE_ENGINES}gen_maxStale${MAX_STALENESS_STEPS}"

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo_fully_async \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.fully_async.max_staleness_steps=${MAX_STALENESS_STEPS} \
  trainer.fully_async.num_parallel_generation_workers=${NUM_PARALLEL_GENERATION_WORKERS} \
  trainer.algorithm.off_policy_correction.sequence_mask_metric=$SEQUENCE_MASK_METRIC \
  trainer.algorithm.off_policy_correction.geo_mask_high=$GEO_MASK_HIGH \
  trainer.algorithm.off_policy_correction.geo_mask_low=$GEO_MASK_LOW \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=false \
  trainer.strategy=megatron \
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_POLICY_GPUS \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.epochs=10 \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=50 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=5 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3_5_dapo_async" \
  trainer.run_name="$RUN_NAME" \
  trainer.export_path="/mnt/local_storage/exports/$RUN_NAME" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="/mnt/local_storage/ckpts/$RUN_NAME" \
  $@
