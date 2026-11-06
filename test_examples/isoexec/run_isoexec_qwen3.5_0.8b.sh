#!/usr/bin/env bash
set -euo pipefail

# Linux/H100; activate an environment with SkyRL, IsoExec, Megatron and vLLM.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ISOEXEC_ROOT="${ISOEXEC_ROOT:-$SKYRL_ROOT/../IsoExec}"
PYTHON_EXECUTABLE="${ISOEXEC_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
PP="${PP:-1}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
RUN_DIR="${RUN_DIR:-${TMPDIR:-/tmp}/skyrl-isoexec-qwen08-pp${PP}-$(date +%Y%m%d-%H%M%S)}"
case "$PP" in
  1|2) NUM_GPUS=$((4 * PP)) ;;
  *) printf 'Use PP=1 (4 GPUs) or PP=2 (8 GPUs).\n' >&2; exit 2 ;;
esac

cd "$SKYRL_ROOT"
export PYTHONPATH="$SKYRL_ROOT:$SKYRL_ROOT/skyrl-gym:$ISOEXEC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export SKYRL_PYTHONPATH_EXPORT=1 SKYRL_LD_LIBRARY_PATH_EXPORT=1
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S="${SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S:-1200}"
runner=(uv run --isolated --no-project "$PYTHON_EXECUTABLE")
# IsoExec selects the shared architecture and reads dimensions from the HF checkpoint.
source_config='{"source":{"package":"isoexec.models.qwen3_5"},"revision":"2fc06364715b967f1860aea9cf38778875588b17","trainer_placement":{"leaves":8,"leaf_dtype":"bf16"},"inference_placement":{"leaves":8,"leaf_dtype":"bf16"}}'
engine_kwargs='{"compilation_config":{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"},"max_model_len":2048,"max_num_seqs":8,"prefill_context_parallel_size":1,"decode_context_parallel_size":1,"quantization":null,"speculative_config":null,"seed":42}'

command=(-m skyrl.train.entrypoints.main_base
  "data.train_data=[\"$DATA_DIR/train.parquet\"]"
  "data.val_data=[\"$DATA_DIR/validation.parquet\"]"
  enable_isoexec=true
  trainer.rollout_logprob_comparison=full
  generator.inference_engine.logprob_output=full
  "isoexec_model=$source_config"
  trainer.strategy=megatron trainer.bf16=true
  trainer.policy.model.path=Qwen/Qwen3.5-0.8B
  trainer.policy.model.lora.rank=0
  trainer.policy.model.fake_int4_qat.enabled=false
  trainer.policy.use_torch_compile=false
  trainer.policy.language_model_only=true
  trainer.policy.megatron_config.tensor_model_parallel_size=2
  "trainer.policy.megatron_config.pipeline_model_parallel_size=$PP"
  trainer.policy.megatron_config.context_parallel_size=1
  trainer.policy.megatron_config.expert_model_parallel_size=1
  trainer.policy.megatron_config.expert_tensor_parallel_size=1
  trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel=true
  trainer.policy.megatron_config.transformer_config_kwargs.deterministic_mode=true
  trainer.policy.megatron_config.fp8=null
  trainer.policy.megatron_config.fp8_param=null
  trainer.placement.colocate_all=true
  trainer.placement.policy_num_nodes=1
  "trainer.placement.policy_num_gpus_per_node=$NUM_GPUS"
  trainer.gradient_checkpointing=true
  trainer.remove_microbatch_padding=true
  trainer.fused_lm_head_logprob=false
  trainer.recompute_old_logprobs_per_minibatch=true
  trainer.algorithm.advantage_estimator=grpo
  trainer.algorithm.use_kl_loss=false
  trainer.algorithm.use_kl_in_reward=false
  trainer.algorithm.kl_loss_coef=0
  trainer.algorithm.dynamic_sampling.type=null
  trainer.train_batch_size=4 trainer.policy_mini_batch_size=4
  trainer.micro_train_batch_size_per_gpu=1 trainer.micro_forward_batch_size_per_gpu=1
  trainer.update_epochs_per_batch=1 trainer.epochs=1 trainer.max_training_steps=1
  trainer.max_prompt_length=512
  trainer.eval_before_train=false trainer.eval_interval=-1
  trainer.ckpt_interval=-1 trainer.hf_save_interval=-1 trainer.resume_mode=null
  trainer.policy.optimizer_config.lr=1e-6
  trainer.policy.optimizer_config.num_warmup_steps=0
  trainer.logger=console
  trainer.project_name=qwen3.5-35b-dapo
  "trainer.run_name=isoexec-qwen3.5-0.8b-pp$PP"
  "trainer.log_path=$RUN_DIR/logs"
  "trainer.ckpt_path=$RUN_DIR/checkpoints"
  "trainer.export_path=$RUN_DIR/exports"
  # MTP/speculative decoding remains off; support is deferred.
  trainer.mtp.enabled=false
  generator.inference_engine.speculative_config=null
  generator.inference_engine.language_model_only=true
  generator.inference_engine.backend=vllm
  generator.inference_engine.run_engines_locally=true
  generator.inference_engine.distributed_executor_backend=mp
  generator.inference_engine.weight_sync_backend=nccl
  generator.inference_engine.num_engines=2
  generator.inference_engine.tensor_parallel_size=2
  "generator.inference_engine.pipeline_parallel_size=$PP"
  generator.inference_engine.data_parallel_size=1
  generator.inference_engine.expert_parallel_size=1
  generator.inference_engine.enforce_eager=false
  generator.inference_engine.enable_prefix_caching=false
  generator.inference_engine.max_num_batched_tokens=1024
  generator.inference_engine.gpu_memory_utilization=0.2
  "generator.inference_engine.engine_init_kwargs=$engine_kwargs"
  generator.batched=true generator.n_samples_per_prompt=2
  generator.sampling_params.temperature=1.0
  generator.sampling_params.top_p=1.0
  generator.sampling_params.max_generate_length=1024
  generator.eval_sampling_params.max_generate_length=1024
  generator.sampling_params.logprobs=1
  environment.env_class=gsm8k
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${runner[@]}" "${command[@]}" "$@"
  printf '\n'
  exit 0
fi
mkdir -p "$(dirname -- "$RUN_DIR")"
mkdir "$RUN_DIR"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/validation.parquet" ]]; then
  "${runner[@]}" examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR" \
    2>&1 | tee "$RUN_DIR/data-preparation.log"
fi
"${runner[@]}" "${command[@]}" "$@" 2>&1 | tee "$RUN_DIR/training.log"
