set -x

# ==========================================================================================
# BITWISE ZERO-KL DAPO for allenai/OLMoE-1B-7B-0924 (MoE) on the NIGHTLY (no-TE) stack
# ==========================================================================================
# MoE twin of run_megatron_dapo_mimo_7b_zerokl_nightly.sh. Same stack, same TP=PP=EP=1 topology,
# same rollout-accel gates. What the MoE path adds (all in zerokl/moe_batch_invariant.py, applied
# identically to the trainer GPTModel and the in-vLLM engine GPTModel):
#   * SequentialMLP (moe_grouped_gemm=false): each expert is a plain F.linear, so vLLM's
#     batch-invariant aten override makes an expert's per-token output independent of how many
#     tokens routed to it. Grouped GEMM tiles over the per-expert token counts -> batch-variant.
#   * allgather token dispatcher: at TP=EP=1 its dispatch/combine collectives are no-ops.
#   * fp32 router, every MoE fusion off, no token dropping / capacity padding.
#   * deterministic expert combine: megatron-core sums the top-8 expert outputs with a CUDA
#     scatter_add_ (atomicAdd over duplicate indices -> arbitrary summation order). Replaced with a
#     fixed ascending-expert-order gather+add. Gate: SKYRL_ZEROKL_MOE_DETERMINISTIC (default 1;
#     set 0 to A/B the batch-variant baseline).
#   * sorted router top-k: megatron-core passes sorted=torch.is_grad_enabled(), so engine (no_grad)
#     and trainer (grad) can order the top-k differently. Harmless for OLMoE (pre_softmax=True) but
#     forced sorted anyway -- see the audit in MOE_ZEROKL_REPORT.md.
#
# Expert weights: megatron-bridge's OLMoE mapping names only the grouped-GEMM params, so the bridge
# is retargeted onto experts.local_experts.N.linear_fcX.weight before it is constructed. Watch for
# zero [ZEROKL-MISS] lines and a coherent first completion -- gibberish means the mapping missed.
#
# GPUs: DP8 / TP1, optimizer CPU-offloaded, same footprint as the MiMo-7B run (7B total params).
# If the headnode is busy, drop NUM_GPUS_PER_NODE + NUM_INFERENCE_ENGINES together. NEVER `ray stop`.
#
# Launch:
#   WANDB_API_KEY=<key> bash examples/zerokl/run_megatron_dapo_olmoe_1b7b_zerokl_nightly.sh \
#       > /mnt/local_storage/logs/zerokl_nightly_dapo_olmoe.log 2>&1
#
# Gate to check: wandb policy/rollout_train_logprobs_abs_diff_mean <= 1e-6 at EVERY step, including
# steps 2-5 (post-weight-update). A clean step 1 with a nonzero step 2 is the sleep/wake weight
# clobber class of bug, not an MoE bug -- set SKYRL_ZEROKL_DEBUG=1 and compare the
# [ZEROKL-REAPPLY] == [SENDER] == [ZEROKL-ENGFWD] checksums.

MODEL_NAME="${OLMOE_MODEL_PATH:-/mnt/local_storage/models/OLMoE-1B-7B-0924}"
DATA_DIR="/mnt/local_storage/data"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
NUM_NODES=1
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1
LOGGER="wandb"

# ----- DAPO knobs (IDENTICAL to the MiMo-7B zero-KL reference) -----
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=16
ENFORCE_EAGER=false
LR=1e-6

# ----- parallelism: TP=PP=CP=EP=1 (DP=8). EP>1 would move the expert combine into a collective. -----
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=null

# ----- MoE config. force_zerokl_moe_config re-pins these last (after transformer_config_kwargs),
# so they cannot be overridden away; set here too so the intent is visible in the launch command.
MOE_TOKEN_DISPATCHER="allgather"
MOE_GROUPED_GEMM=false
MOE_ROUTER_DTYPE="fp32"

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

REMOVE_MICROBATCH_PADDING=false

# ===== zero-KL switches =====
export SKYRL_ZERO_KL=1
export SKYRL_ZEROKL_LOCAL_SPEC=1
export SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=1
export SKYRL_ZEROKL_MOE_DETERMINISTIC=1    # fixed-order expert combine + sorted router top-k
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VARLEN_FORCE_NUM_SPLITS_1=1
# Rollout accel: validated bitwise-safe under the num_splits=1 CUSTOM varlen backend on the dense
# path (4.6x generate). Re-validate on MoE: if step-1 abs_diff is nonzero, drop these three and set
# SKYRL_ZEROKL_NO_CHUNKED_PREFILL=1 + ENFORCE_EAGER=true to isolate.
export SKYRL_ZEROKL_ENABLE_PREFIX_CACHE=1
export SKYRL_ZEROKL_ENABLE_CHUNKED_PREFILL=1
export SKYRL_ZEROKL_ENABLE_CUDAGRAPH=1
export SKYRL_ZEROKL_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export SKYRL_ZEROKL_FWD_PROBE=0
export SKYRL_ZEROKL_BISECT=0
export SKYRL_ZEROKL_SEQ_PROBE=0
DISTRIBUTED_EXECUTOR_BACKEND="mp"

uv run --isolated --extra zerokl -m examples.train.algorithms.dapo.main_dapo \
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
  trainer.policy.megatron_config.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  trainer.policy.megatron_config.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  trainer.policy.megatron_config.moe_router_dtype=$MOE_ROUTER_DTYPE \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.algorithm.off_policy_correction.tis_ratio_type=null \
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
  generator.inference_engine.gpu_memory_utilization=0.42 \
  trainer.logger="$LOGGER" \
  trainer.project_name="olmoe_1b7b_dapo" \
  trainer.run_name="zerokl_nightly_dapo_olmoe_1b7b_dp${NUM_GPUS_PER_NODE}" \
  trainer.export_path="/mnt/local_storage/exports/zerokl_nightly_dapo_olmoe_1b7b" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="/mnt/local_storage/ckpts/zerokl_nightly_dapo_olmoe_1b7b" \
  "$@"
