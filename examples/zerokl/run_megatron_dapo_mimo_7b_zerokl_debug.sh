set -x

# FAST DEBUG variant of run_megatron_dapo_mimo_7b_zerokl.sh.
# Purpose: iterate on zero-KL correctness (GPTModel-in-vLLM forward + native weight sync) WITHOUT
# the 25-min generation. Changes vs the full script:
#   * MAX_RESPONSE_LENGTH=256, MAX_PROMPT_LENGTH=512  -> even gibberish finishes fast
#   * tiny batch (8 prompts x 4 samples = 32 seqs)
#   * SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=1  -> engine GPTModel loads real MiMo via bridge at init
#       (separates "is the forward correct" from "is the sync correct"; with load=0 the bridge may
#        leave params on meta/uninitialized so native-sync copy_ has nothing to write -> garbage).
#   * LOGGER=console  -> no wandb noise for debug
# Watch the driver log for:
#   [ZEROKL-SYNC] target GPTModel has N params ... sample target names: [...]
#   [ZEROKL-SYNC] MISS (sender name not in target): <name>      <- name mismatch (the bug)
#   [ZEROKL-SYNC] running copied=<C> missed=<M>                 <- last line = final counts
#   reward/avg_raw_reward + the sampled responses (coherent vs gibberish)

MODEL_NAME="XiaomiMiMo/MiMo-7B-RL"
DATA_DIR="/mnt/local_storage/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
NUM_NODES=1
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1
LOGGER="console"

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=false
OVERLONG_BUFFER_LEN=$((256))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=256
TRAIN_BATCH_SIZE=8
MINI_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=4
EVAL_N_SAMPLES_PER_PROMPT=1
ENFORCE_EAGER=true
LR=1e-6

MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=null
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0
REMOVE_MICROBATCH_PADDING=false

export SKYRL_ZERO_KL=1
# DEBUG: enable the per-sync / per-step diagnostic checksums + probe prints (SENDER/RECEIVER/
# ENGFWD/SCOREFWD/POSTSTEP abs-sums). OFF by default in production (each is a full-model fp64
# reduction; ~4/sync + 2/optim-step) -- gated on this flag so the production run skips them.
export SKYRL_ZEROKL_DEBUG=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# DEBUG: load real MiMo weights into the engine GPTModel at init (bridge), so generation
# coherence tests the FORWARD; the [ZEROKL-SYNC] prints separately test the native sync.
export SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=1
# BISECT: three-way weight checksum (SENDER vs TRAINER-forward vs ENGINE-runtime) to localize the
# multi-process 0.0104. Single-process checks proved model+wrapper+init-weights all bitwise.
export SKYRL_ZEROKL_BISECT=1
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
DISTRIBUTED_EXECUTOR_BACKEND="mp"

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
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
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.algorithm.off_policy_correction.tis_ratio_type=null \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.epochs=1 \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100000 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
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
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mimo_7b_rl_dapo" \
  trainer.run_name="zerokl_dapo_DEBUG" \
  trainer.ckpt_interval=-1 \
  trainer.resume_mode=none \
  $@
