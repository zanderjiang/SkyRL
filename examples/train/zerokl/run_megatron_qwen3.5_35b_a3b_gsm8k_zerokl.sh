set -x

# ==========================================================================================
# ZERO-KL GRPO for Qwen/Qwen3.5-35B-A3B-Base on GSM8K with Megatron.
# ==========================================================================================
# Zero-KL twin of examples/train/megatron/run_megatron_qwen3.5_35b_a3b.sh. The rollout engine
# runs Megatron's GPTModel inside vLLM (skyrl.backends.skyrl_train.zerokl.gptmodel_vllm) and
# weights are synced NATIVELY (no HF conversion), so the rollout token logprobs and the trainer's
# forward are (near-)bitwise identical and the importance ratio is ~1 without TIS.
#
# Everything task-side (GRPO, GSM8K data, seq lengths, batch sizes, lr, ckpt) is kept IDENTICAL to
# the non-zerokl gsm8k reference so the two are directly comparable on reward. The differences are
# ONLY the zero-KL machinery:
#   * SKYRL_ZERO_KL=1 (+ MoE determinism)  -> unified GPTModel on both ends + native weight sync
#   * MATCHED TP on both ends              -> Megatron TP == inference TP (== 8 here). This is the
#                                             core zero-KL requirement: the trainer forward and the
#                                             in-vLLM GPTModel forward must run the same TP layout.
#                                             (The TP=1 validation scripts were just the first
#                                             matched setting; matched TP>1 is the intended path.)
#   * EP=1 (ETP=8) + allgather dispatcher  -> deterministic expert combine. EP>1 turns the top-k
#                                             expert combine into a cross-rank collective whose
#                                             summation order is nondeterministic -> breaks bitwise.
#   * TP=8 (DP=1)                           -> shards the 35B weights ~8.75GB/GPU/copy so a full
#                                             trainer + a full in-vLLM GPTModel fit colocated on 80GB
#                                             (TP=1 cannot: ~70GB x2 > 80GB).
#   * async_engine=false + in-process vLLM -> the GPTModel string-registration done in the engine
#                                             actor must reach the model build (no fresh mp worker).
#   * tis_ratio_type=null                  -> TIS off (unnecessary at zero KL).
#   * --extra zerokl / run_name / paths    -> zerokl venv + zerokl_* so it does not collide.
#
# GDN: 3 of every 4 Qwen3.5 layers are GatedDeltaNet linear attention. Their decode-vs-prefill
# divergence is FIXED -- chunk-consistent decode makes the GDN layers bitwise (65536/65536 tokens
# exact on Qwen3.5-0.8B), and the whole model is bitwise through the GPTModel-in-vLLM engine
# (256/256, max 0.0). See examples/zerokl/nightly/GDN_ZEROKL_REPORT.md. SKYRL_ZEROKL_GDN=1 turns on:
#   * the `fla` shim, so the Megatron trainer runs the engine's GDN ops (no TransformerEngine, no
#     flash-linear-attention),
#   * the hybrid no-TE layer spec + the Qwen3.5 bridge mapping retarget,
#   * chunk-consistent GDN decode in the engine.
#
# NOT YET VALIDATED AT THIS SCALE. Everything above was proven on Qwen3.5-0.8B dense-MoE-free hybrid
# at TP=1. This run is the first to exercise (a) TP=8 GDN, and (b) the MoE branch of the hybrid spec
# and of the bridge mapping. The zero-KL gate is unchanged: watch
# policy/rollout_train_logprobs_abs_diff_mean -- it must be <= 1e-6 at EVERY step, including 2..5
# after the first weight sync, and policy_kl must be 0.0. If step 1 is clean and step 2 is not, that
# is the sleep/wake weight-clobber class of bug, not GDN.
#
# CAVEAT (throughput): chunk-consistent decode costs ~5.8x rollout tok/s at 16 concurrent sequences
# (measured; it is the per-slot python loop in ChunkConsistentGDN.decode, not FLOPs). Expect a slow
# rollout until that loop is batched.
#
# Runs on 1 node of 8xH100s (80GB each). Throughput WILL be lower than the baseline (batch-invariant
# kernels + eager + unified model + optimizer offload) -- expected; we are measuring KL, not speed.
#
# Prepare data first (same as the baseline):
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# Launch:
#   WANDB_API_KEY=<key> bash examples/train/zerokl/run_megatron_qwen3.5_35b_a3b_gsm8k_zerokl.sh \
#       > /mnt/local_storage/logs/zerokl_gsm8k_qwen3.5_35b.log 2>&1

DATA_DIR="$HOME/data/gsm8k"
LOGGER="${ZEROKL_LOGGER:-wandb}"  # ZEROKL_LOGGER=console prints step metrics to stdout (gate validation)
MODEL_NAME="Qwen/Qwen3.5-35B-A3B-Base"

INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron

NUM_NODES=1
NUM_GPUS=8

# ----- parallelism: MATCHED TP=8 on both ends; EP=1 (ETP=8) for deterministic expert combine -----
# Megatron TP MUST equal inference TP for zero-KL (same forward layout on trainer and rollout).
MEGATRON_TP=4    # smallest TP that fits trainer+engine colocated (17.5+17.5 GB weights); DP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1     # EP=1 is required: EP>1 makes the expert combine a nondeterministic collective.
MEGATRON_ETP=4    # experts are tensor-parallel sharded across the TP group (ETP == TP at EP=1).

NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=4   # == MEGATRON_TP (matched); 2 engines generate in parallel

# ----- MoE determinism config (re-pinned by force_zerokl_moe_config; set here so intent is visible) -----
MOE_TOKEN_DISPATCHER="allgather"   # at EP=1 the dispatch/combine collectives are no-ops
MOE_GROUPED_GEMM=false             # SequentialMLP: each expert a plain F.linear -> batch-invariant
MOE_ROUTER_DTYPE="fp32"

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Sequence lengths (kept identical to the gsm8k reference).
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024

# THD sample packing OFF for zero-KL bring-up (matches the MoE zero-KL precedent): packing changes
# the microbatch token layout and complicates the bitwise rollout==train comparison. A/B to true
# only after the abs_diff gate passes.
REMOVE_MICROBATCH_PADDING=false

# Eager for bring-up; CUDA graphs are gated OFF for MoE via SKYRL_ZEROKL_ENABLE_CUDAGRAPH below.
ENFORCE_EAGER=true

# Qwen3.5 flags
LANGUAGE_MODEL_ONLY=True # qwen3-vl in megatron has a separate sequence packing path - if using language_model_only, use the native GPTModel + GDN thd packing path
ENGINE_INIT_KWARGS='{"gdn_prefill_backend": "triton"}' # see https://github.com/vllm-project/vllm/issues/36921#issuecomment-4109702738
DISTRIBUTED_EXECUTOR_BACKEND="mp"

# ===== zero-KL switches =====
export SKYRL_ZERO_KL=1
export SKYRL_ZEROKL_LOCAL_SPEC=1
# 0: the engine GPTModel is populated by the FIRST NATIVE SYNC (which precedes the first rollout),
# not by an HF disk load at init. At 35B, =1 would have all 8 TP workers stream the 70GB HF
# checkpoint through host RAM at engine build for weights that are immediately overwritten.
export SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=0
export SKYRL_ZEROKL_MOE_DETERMINISTIC=1     # fixed-order expert combine + sorted router top-k
export SKYRL_ZEROKL_GDN=1                   # fla shim (trainer) + hybrid no-TE spec + chunk-consistent decode (engine)
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0     # in-process vLLM so GPTModel registration reaches the model build
export VARLEN_FORCE_NUM_SPLITS_1=1
# Chunk-consistent GDN decode redefines ssm_state[slot] as the state at the last CHUNK BOUNDARY.
# Prefix caching and chunked prefill both read/resume that state on their own terms, so they must be
# OFF (the engine raises rather than degrade quietly -- gdn_engine_patch.assert_engine_args_compatible).
# They ARE bitwise-safe for the softmax layers and worth 4.6x rollout; supporting them for GDN is a
# follow-up, not a bug. Same for CUDA graphs.
export SKYRL_ZEROKL_ENABLE_PREFIX_CACHE=0
export SKYRL_ZEROKL_ENABLE_CHUNKED_PREFILL=0
export SKYRL_ZEROKL_ENABLE_CUDAGRAPH=0      # OFF for MoE bring-up (SequentialMLP needs a per-layer d2h sync)
# With chunked prefill off, vLLM requires max_num_batched_tokens >= max_model_len; this is the flag
# that caps max_model_len and bumps the token budget accordingly (vllm_engine.setup_envvars_for_vllm).
export SKYRL_ZEROKL_NO_CHUNKED_PREFILL=1
export SKYRL_ZEROKL_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# vLLM's Qwen3.5 module imports the VL chain unconditionally; torchvision is absent from the zerokl
# venv. The stub only has to satisfy the import. (Same as the 0.8B nightly launcher.)
export PYTHONPATH="$(cd "$(dirname "$0")/../../zerokl/nightly/_torchvision_stub" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/mnt/local_storage/hf}"

# On Blackwell, use the following env vars:
# export VLLM_USE_FLASHINFER_MOE_FP16=0   # force triton moe backend since flashinfer trtllm bf16 MoE kernel requires expert intermediate_size to be a multiple of 128
# export FLA_TILELANG=0   # force triton gdn backend since fla's default TileLang GDN backend aborts in the packed backward. leave unset on hopper, since Triton GDN backward is broken there: https://github.com/fla-org/flash-linear-attention/issues/640#issuecomment-4236520788

uv run --isolated --extra zerokl -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
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
  trainer.algorithm.off_policy_correction.tis_ratio_type=null \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.logger="$LOGGER" \
  trainer.project_name="${ZEROKL_WANDB_PROJECT:-gsm8k_qwen3.5}" \
  trainer.run_name="zerokl_gsm8k_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}_qwen3.5-35b-a3b" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/zerokl_gsm8k_megatron_ckpt" \
  $@
