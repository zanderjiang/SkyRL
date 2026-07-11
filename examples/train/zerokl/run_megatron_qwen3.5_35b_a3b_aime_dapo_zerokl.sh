set -x

# ==========================================================================================
# ZERO-KL DAPO for Qwen/Qwen3.5-35B-A3B-Base on AIME (dapo-math-17k) with Megatron.
# ==========================================================================================
# The long-response showcase run: DAPO on AIME stacks the factors where trainer/rollout mismatch
# actually bites (8K responses -- mismatch grows with length; sparse hard rewards; TIS off), so the
# zero-KL benefit is measurable against a non-zerokl baseline, unlike GSM8K where both saturate.
#
# Zero-KL machinery is copied VERBATIM from run_megatron_qwen3.5_35b_a3b_gsm8k_zerokl.sh (which
# validated the 35B MoE+GDN TP8 stack live: rollout_train_logprobs_abs_diff_mean ~5.5e-7 for 21+
# steps). Task side follows the MiMo-7B DAPO nightly recipe (examples/zerokl/
# run_megatron_dapo_mimo_7b_zerokl_nightly.sh) with 2K prompt / 8K response.
#
# Batching: train_batch == mini_batch = 32 -> exactly ONE optimizer update per step, so every
# trained token is exactly on-policy under zero-KL (no second-minibatch staleness).
#
# THROUGHPUT WARNING: 1 engine x TP8 with chunk-consistent GDN decode (~5.8x slower decode) at 8K
# generations means LONG steps (order 1-3h). We are measuring reward-per-step vs baseline at
# matched updates, not speed.
#
# Zero-KL gate (unchanged): policy/rollout_train_logprobs_abs_diff_mean <= 1e-6 at EVERY step.
#
# Launch:
#   WANDB_API_KEY=<key> bash examples/train/zerokl/run_megatron_qwen3.5_35b_a3b_aime_dapo_zerokl.sh \
#       > /mnt/local_storage/logs/zerokl_dapo_aime_35b.log 2>&1

DATA_DIR="/mnt/local_storage/data"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER="${ZEROKL_LOGGER:-wandb}"
MODEL_NAME="Qwen/Qwen3.5-35B-A3B-Base"

INFERENCE_BACKEND="vllm"

NUM_NODES=1
NUM_GPUS=8

# ----- parallelism: MATCHED TP=8 both ends; EP=1 (ETP=8) for deterministic expert combine -----
# (See the gsm8k zerokl script header for the full rationale: matched TP is the core zero-KL
# requirement; EP>1 makes the expert combine a nondeterministic collective; TP=8 is the memory
# sweet spot for colocated trainer + in-vLLM GPTModel on 80GB.)
MEGATRON_TP=8
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=8

NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=8   # == MEGATRON_TP (matched)

# ----- MoE determinism config -----
MOE_TOKEN_DISPATCHER="allgather"
MOE_GROUPED_GEMM=false
MOE_ROUTER_DTYPE="fp32"

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# ----- DAPO knobs (MiMo nightly parity) -----
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7

# Sequence lengths: AIME reasoning needs >2K; 8K matches the MiMo DAPO reference.
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))

REMOVE_MICROBATCH_PADDING=false
ENFORCE_EAGER=true

# Qwen3.5 flags
LANGUAGE_MODEL_ONLY=True
ENGINE_INIT_KWARGS='{"gdn_prefill_backend": "triton"}'
DISTRIBUTED_EXECUTOR_BACKEND="mp"

# ===== zero-KL switches (identical to the validated 35B gsm8k zerokl run) =====
export SKYRL_ZERO_KL=1
export SKYRL_ZEROKL_LOCAL_SPEC=1
export SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=0
export SKYRL_ZEROKL_MOE_DETERMINISTIC=1
export SKYRL_ZEROKL_GDN=1
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VARLEN_FORCE_NUM_SPLITS_1=1
export SKYRL_ZEROKL_ENABLE_PREFIX_CACHE=0
export SKYRL_ZEROKL_ENABLE_CHUNKED_PREFILL=0
export SKYRL_ZEROKL_ENABLE_CUDAGRAPH=0
export SKYRL_ZEROKL_NO_CHUNKED_PREFILL=1
export SKYRL_ZEROKL_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export SKYRL_ZEROKL_DEBUG=1   # keep the per-step checksum/DIFF probes observable (gate validation)
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export PYTHONPATH="$(cd "$(dirname "$0")/../../zerokl/nightly/_torchvision_stub" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/mnt/local_storage/hf}"

uv run --isolated --extra zerokl -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.off_policy_correction.tis_ratio_type=null \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  trainer.policy.megatron_config.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  trainer.policy.megatron_config.moe_router_dtype=$MOE_ROUTER_DTYPE \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.epochs=10 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.logprobs_chunk_size=256 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_ckpts_to_keep=2 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.num_warmup_steps=5 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=16 \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.logger="$LOGGER" \
  trainer.project_name="${ZEROKL_WANDB_PROJECT:-qwen3_5_35b_dapo_aime}" \
  trainer.run_name="zerokl_dapo_aime_tp${MEGATRON_TP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}_qwen3.5-35b-a3b" \
  trainer.resume_mode=null \
  trainer.ckpt_path="/mnt/local_storage/ckpts/zerokl_dapo_aime_35b" \
  $@
