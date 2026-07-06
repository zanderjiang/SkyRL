set -x

# ==========================================================================================
# TRUE BITWISE ZERO-KL DAPO for XiaomiMiMo/MiMo-7B-RL on the NIGHTLY (no-TransformerEngine) stack
# ==========================================================================================
# This is the bitwise-zero-KL twin of run_megatron_dapo_mimo_7b_zerokl.sh. The difference from that
# (near-zero, production TE stack) script is the STACK + a few switches:
#   * `uv run --extra zerokl`  -> torch-2.14.dev + vLLM-1.0.dev + megatron-core LOCAL spec, NO TE.
#     The Anyscale Ray uv-runtime-env hook propagates this env to every actor.
#   * SKYRL_ZEROKL_LOCAL_SPEC=1 -> trainer GPTModel (megatron_worker) AND engine GPTModel
#     (gptmodel_vllm) build with Megatron's LOCAL layer spec (plain torch SDPA/RMSNorm/F.linear),
#     and the engine selects the CUSTOM num_splits=1 varlen attention backend -> bitwise
#     decode==prefill (validated 256/256 max==0 by examples/zerokl/nightly/skyrl_engine_parity_test.py).
#   * SKYRL_ZEROKL_TRAINER_PATCHES is NOT set -> the TE-targeted batch-invariant trainer patches are
#     skipped (they require TE, which is absent here); batch invariance comes from VLLM_BATCH_INVARIANT.
#   * TIS OFF (tis_ratio_type=null) -- genuinely unnecessary at bitwise zero-KL.
#
# GPUs: only 4 are used (the headnode often shares GPUs 0-3 with another job). Ray's placement group
# picks the free GPUs. 4 engines (DP=4) + optimizer CPU offload keeps host RAM well under the 1999GB
# headnode budget (the 8-engine config previously host-OOM'd -- see memory zerokl-headnode-oom).
#
# Data already prepared at /mnt/local_storage/data (dapo-math-17k-cleaned + aime-2024-cleaned).
# Launch:
#   WANDB_API_KEY=<key> bash examples/zerokl/run_megatron_dapo_mimo_7b_zerokl_nightly.sh \
#       > /mnt/local_storage/logs/zerokl_nightly_dapo_mimo_7b.log 2>&1

MODEL_NAME="/mnt/local_storage/models/MiMo-7B-RL"
DATA_DIR="/mnt/local_storage/data"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
NUM_NODES=1
# 8 GPUs = DP8 (8 engines + 8 TP1 trainer replicas). The trainer MUST stay TP1 to match the engine's
# TP1 bitwise (TP>1 changes the all-reduce order -> not bitwise). DP8 ~2x the 4-GPU throughput; host
# RAM for the 8x CPU-offloaded optimizer (~670GB) fits with headroom (watchdog guards the headnode).
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1
LOGGER="wandb"

# ----- DAPO knobs (IDENTICAL to the non-zerokl reference) -----
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
# Full baseline-scale response length. 2048 was a first-validation value, but it truncated EVERY AIME
# rollout (avg_response_length pinned at the cap, avg_raw_reward stuck at the -2 overlong floor,
# pass@8~0) -> no reward signal -> entropy collapse + chaotic loss. AIME reasoning needs >2K tokens, so
# use 8K (matches the non-zerokl reference). SKYRL_ZEROKL_MAX_MODEL_LEN below auto-follows (= prompt +
# response = 10K). NOTE: longer KV raises GPU+host RAM; keep the headnode watchdog running (see memory
# zerokl-headnode-oom) and drop gpu_memory_utilization / engine count if it approaches the RAM ceiling.
MAX_RESPONSE_LENGTH=$((1024 * 8))
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=16
ENFORCE_EAGER=true
LR=1e-6

# ----- parallelism: TP=1 (DP=4) -----
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=null

# ----- optimizer offload ON: fit 7B + Adam at TP=1 on one 80GB GPU -----
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

REMOVE_MICROBATCH_PADDING=false

# ===== zero-KL switches =====
export SKYRL_ZERO_KL=1
export SKYRL_ZEROKL_LOCAL_SPEC=1            # NIGHTLY: local (no-TE) layer spec + CUSTOM varlen backend
export SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=1   # engine loads HF->local weights at build (bridge)
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VARLEN_FORCE_NUM_SPLITS_1=1
# Trainer attention = plain non-paged varlen_attn (num_splits=1, window=(-1,0)). The paged/varlen_out
# experiments were a red herring: the ~0.014 residual was the trainer runtime never activating torch's
# FA3 flash-attn impl (the engine's varlen_backend does; the trainer builds no vLLM engine so it stayed
# unset -> a different kernel). swap_trainer_core_attention_varlen now calls activate_flash_attention_impl
# ("FA3"), so plain varlen_attn is bitwise == engine (verified probe_attn_variants.py: 0/2048). PAGED unset.
# Force chunked prefill OFF for bitwise (a chunk-split prompt would feed the varlen kernel partial
# KV). Requires max_num_batched_tokens >= max_model_len; we cap max_model_len = prompt+response.
export SKYRL_ZEROKL_NO_CHUNKED_PREFILL=1
export SKYRL_ZEROKL_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
# Legacy in-process engine path (the zero-KL GPTModel registration hook lives there).
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# NOTE: SKYRL_ZEROKL_TRAINER_PATCHES intentionally UNSET (TE patches skipped on the no-TE stack).
# Diagnostic: localize the residual -> trainer-machinery (padding/Float16Module/forward_backward_func)
# via [ZEROKL-EXTRACT] (from_parallel vs plain log_softmax) and [ZEROKL-FWDPROBE] (bare unpadded
# GPTModel vs fbf result). Both print from the trainer worker (whose stdout IS forwarded).
# Diagnostics OFF for the production run (localization complete: the residual is the vLLM-paged vs
# Megatron-non-paged attention-path bf16 drift, ~0.014 mean, handled by TIS below). Flip any to 1 to
# re-enable: FWD_PROBE (bare-vs-fbf), BISECT (weight fingerprints), SEQ_PROBE (dump tokens for rescore).
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
  trainer.project_name="mimo_7b_rl_dapo" \
  trainer.run_name="zerokl_nightly_dapo_mimo_7b_rl_dp${NUM_GPUS_PER_NODE}" \
  trainer.export_path="/mnt/local_storage/exports/zerokl_nightly_dapo_mimo_7b_rl" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="/mnt/local_storage/ckpts/zerokl_nightly_dapo_mimo_7b_rl" \
  "$@"
