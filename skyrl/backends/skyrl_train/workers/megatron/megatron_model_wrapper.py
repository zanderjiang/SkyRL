import os
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import megatron.core.parallel_state as mpu
import torch
import torch.nn as nn
from megatron.core.distributed import finalize_model_grads
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf

from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    get_model_config,
    make_batch_generator,
    model_packs_sequences_internally,
    preprocess_packed_seqs,
    recover_left_padding,
    remove_left_padding,
)
from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
    vocab_parallel_entropy,
    vocab_parallel_entropy_packed_sequences,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    compute_approx_kl,
)
from skyrl.backends.skyrl_train.utils.replay_utils import (
    setup_per_microbatch_replay_backward,
    setup_per_microbatch_replay_forward,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.backends.skyrl_train.workers.worker_utils import (
    compute_minibatch_rollout_logprob_diff_metrics,
)
from skyrl.train.config import TrainerConfig


def _zerokl_scoring_ctx():
    """SkyRL-ZeroKL: run the Megatron forward/backward under the unified (vops) RMSNorm so the
    trainer's logprobs match the vLLM-GPTModel rollout. Applied to BOTH the no-grad old-logprob
    recompute AND the grad training forward, so old==new (is_ratio==1 at the first inner step,
    SkyRL's per-minibatch-recompute guarantee) AND both match the rollout (rollout_train diff small).
    No-op when SKYRL_ZERO_KL != 1.
    Bisect toggle: SKYRL_ZEROKL_SCORING_FORWARD=0 disables the vops-norm wrap (to test whether it
    is what breaks the trainer's Megatron forward -> near-uniform logits / entropy ~8)."""
    if os.environ.get("SKYRL_ZERO_KL") == "1" and os.environ.get("SKYRL_ZEROKL_SCORING_FORWARD", "1") == "1":
        from skyrl.backends.skyrl_train.zerokl.megatron_patches import scoring_mode

        return scoring_mode()
    return nullcontext()


def _build_packed_targets(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_params,
    sub_seq_lengths: Optional[list[list[int]]] = None,
) -> torch.Tensor:
    """Pack full target token IDs without context-parallel sharding."""
    cu_padded = packed_seq_params.cu_seqlens_q_padded.to(device=sequences.device, dtype=torch.long)
    total_padded_tokens = int(cu_padded[-1].item())

    targets = torch.zeros((total_padded_tokens,), dtype=sequences.dtype, device=sequences.device)
    if sub_seq_lengths is not None:
        cu_padded_cpu = cu_padded.detach().cpu().tolist()
        seg_idx = 0
        for row_idx, row_lens in enumerate(sub_seq_lengths):
            row_offset = 0
            for seq_len in row_lens:
                seq_len = int(seq_len)
                if seg_idx + 1 >= len(cu_padded_cpu):
                    raise ValueError("sub_seq_lengths contains more sub-sequences than packed_seq_params")
                packed_start = cu_padded_cpu[seg_idx]
                targets[packed_start : packed_start + seq_len] = sequences[row_idx, row_offset : row_offset + seq_len]
                row_offset += cu_padded_cpu[seg_idx + 1] - cu_padded_cpu[seg_idx]
                seg_idx += 1
        if seg_idx != len(cu_padded_cpu) - 1:
            raise ValueError(
                f"sub_seq_lengths describes {seg_idx} sub-sequences, "
                f"but packed_seq_params describes {len(cu_padded_cpu) - 1}"
            )
        return targets.unsqueeze(0)

    attention_mask = attention_mask.to(device=sequences.device, dtype=torch.bool)
    token_offsets = attention_mask.to(torch.long).cumsum(dim=1) - 1
    packed_indices = cu_padded[:-1].unsqueeze(1) + token_offsets
    targets[packed_indices[attention_mask]] = sequences[attention_mask]
    return targets.unsqueeze(0)


class MegatronModelWrapper:
    def __init__(
        self,
        config: TrainerConfig,
        actor_module: List[nn.Module],
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_loss_fn: Optional[Callable] = None,
    ):
        self.cfg = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.policy_loss_fn = policy_loss_fn
        self.remove_microbatch_padding = self.cfg.remove_microbatch_padding
        # Some models (e.g. Qwen3.5 via the VL bridge -> Qwen3VLModel) pack
        # sequences inside their own forward; SkyRL sample packing would then
        # double-pack and corrupt the GDN cu_seqlens, so refuse it. For Qwen3.5,
        # use language_model_only=True (native GPTModel GDN path) to pack.
        if self.remove_microbatch_padding and model_packs_sequences_internally(self.actor_module):
            raise ValueError(
                "remove_microbatch_padding=True (sample packing) is not supported for models that "
                "pack sequences inside their own forward (e.g. the Qwen3.5 VL Qwen3VLModel): it "
                "double-packs and corrupts the GatedDeltaNet cu_seqlens. Set "
                "trainer.policy.language_model_only=True to route Qwen3.5 to the native GPTModel GDN "
                "packing path, or set trainer.remove_microbatch_padding=False."
            )

        config = get_model_config(self.actor_module[0])
        # This is set to None by default: https://github.com/NVIDIA/Megatron-LM/blob/07b22a05136a3cb08ece05f7de38cf6aeeb165fb/megatron/core/model_parallel_config.py#L95
        # use the build in finalize_model_grads function to all reduce gradients across parallelism dimensions
        config.finalize_model_grads_func = finalize_model_grads
        # Wire up the optimizer's loss scaler so Megatron's pipeline schedule can scale
        # the loss before backward (critical for fp16 dynamic loss scaling, MoE aux loss
        # scaling, and any explicit loss_scale configuration).
        if actor_optimizer is not None:
            config.grad_scale_func = actor_optimizer.scale_loss

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward-only inference to compute log-probs over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: List of micro-batch dicts with keys: "sequences", "attention_mask", "position_ids",
                           and "num_actions".
            seq_len: Padded sequence length per sample.
            micro_batch_size: Per-micro-batch size.
            temperature: Optional temperature scaling for logits.

        Returns:
            torch.Tensor of concatenated log-probs across micro-batches (valid on pipeline last stage only).
        """
        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, data):
            sequences = data["sequences"]
            packed_seq_params = data.get("packed_seq_params")
            packed_targets = data.get("packed_targets")
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            if temperature != 1.0:
                logits.div_(temperature)

            if packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    group=tp_grp,
                    inference_only=True,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    sequences,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    tp_group=tp_grp,
                    inference_only=True,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
                )
            # SkyRL-ZeroKL probe (A): is the EXTRACTION the residual? Compare Megatron's
            # from_parallel_logits_to_logprobs against the standalone's plain fp32 log_softmax+gather
            # on the SAME logits. If they differ, the extraction method contributes the 0.0094.
            if os.environ.get("SKYRL_ZEROKL_FWD_PROBE") == "1" and packed_seq_params is None:
                with torch.no_grad():
                    _ref_lp = torch.log_softmax(logits.float(), dim=-1)
                    _tgt = sequences[:, 1:]
                    _ref = _ref_lp[:, :-1].gather(-1, _tgt.unsqueeze(-1)).squeeze(-1)
                    _n = min(_ref.shape[1], token_logprobs.shape[1])
                    _d = (_ref[:, -_n:] - token_logprobs[:, -_n:]).abs()
                    print(f"[ZEROKL-EXTRACT] from_parallel vs log_softmax: max={float(_d.max()):.3e} "
                          f"mean={float(_d.mean()):.3e} (logits {tuple(logits.shape)})", flush=True)
            return torch.tensor(0.0, device=token_logprobs.device), {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=get_model_config(model),
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            sub_seq_lengths_field = batch.get("sub_seq_lengths")
            sub_seq_lengths = [t.tolist() for t in sub_seq_lengths_field] if sub_seq_lengths_field is not None else None
            batch["sub_seq_lengths_list"] = sub_seq_lengths

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                    sub_seq_lengths=sub_seq_lengths,
                )
                batch["packed_seq_params"] = packed_seq_params
                batch["packed_targets"] = _build_packed_targets(
                    sequences, attention_mask, packed_seq_params, sub_seq_lengths=sub_seq_lengths
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            # SkyRL-ZeroKL: pass attention_mask=None (pure causal flash) to match the engine's
            # causal-flash path. new_sequences is unpadded + decoder-causal; an explicit mask sends
            # TE down a different flash variant -> diffuse ~0.01 logprob drift vs the engine.
            _zk_mask = None if (os.environ.get("SKYRL_ZERO_KL") == "1" and packed_seq_params is None) else new_attention_mask
            outputs = model(
                new_sequences,
                new_position_ids,
                _zk_mask,
                packed_seq_params=packed_seq_params,
            )

            if not self.remove_microbatch_padding:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

            return outputs, partial(collection_func, data=batch)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        with _zerokl_scoring_ctx():
            output = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=len(micro_batches),
                seq_length=seq_len,
                micro_batch_size=micro_batch_size,
                forward_only=True,
            )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            log_probs = [o["log_probs"] for o in output]
            log_probs = torch.cat(log_probs, dim=0)
            # take last num_actions tokens per micro; concatenate later
            # Assume all micros have same num_actions
            num_actions = micro_batches[0]["num_actions"]
            log_probs = log_probs[:, -num_actions:]
            # Per-tensor TRAINER forward-weight dump (once), in the SCORING forward which is proven to
            # be hit (FWD_PROBE dumps here). The BISECT block at line ~685 is in forward_backward_mini_batch
            # which isn't hit before the metric. Writes the SAME formula as the engine (bf16->fp32->sum)
            # to compare to BISECT-ENGINE 89866863.40, plus per-tensor to /mnt/local_storage/zerokl_trn_w.txt.
            _zk_r0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
            if (os.environ.get("SKYRL_ZEROKL_BISECT") == "1" and _zk_r0
                    and not getattr(self, "_zk_wdump_done", False)):
                self._zk_wdump_done = True
                try:
                    from skyrl.backends.skyrl_train.zerokl.weight_fingerprint import fingerprint_line
                    _gsum, _seen2 = 0.0, set()
                    with open("/mnt/local_storage/zerokl_trn_w.txt", "w") as _wf2:
                        for _mm in self.actor_module:
                            _inn = _mm
                            for _ in range(4):
                                _inn = _inn.module if hasattr(_inn, "module") else _inn
                            for _nm2, _p2 in _inn.named_parameters():
                                if _nm2 in _seen2 or _nm2.startswith("mtp."):
                                    continue
                                _seen2.add(_nm2)
                                _t2 = _p2.detach()
                                if hasattr(_t2, "full_tensor"):
                                    try:
                                        _t2 = _t2.full_tensor()
                                    except Exception:
                                        pass
                                # Same 3-way element-wise fingerprint as the engine dump (bf16 basis)
                                # so a name-by-name compare proves engine-at-gen == trainer-at-score.
                                _wf2.write(fingerprint_line(_nm2, _t2))
                                _gsum += float(_t2.to(torch.bfloat16).float().double().abs().sum())
                    with open("/mnt/local_storage/zerokl_probe.log", "a") as _pf2:
                        _pf2.write(f"BISECT-TRAINER cksum={_gsum:.6f} n={len(_seen2)}\n")
                    print("[ZEROKL-WDUMP] trainer scoring-fwd weights -> zerokl_trn_w.txt", flush=True)
                except Exception as _e2:
                    print(f"[ZEROKL-WDUMP] trainer dump failed: {_e2}", flush=True)
            # SkyRL-ZeroKL probe (B): is the forward MACHINERY the residual? Run the standalone's
            # EXACT forward (bare GPTModel, unpadded single sequence, scoring_mode, attention_mask=None,
            # plain log_softmax) on micro_batch 0 and compare to the forward_backward_func result.
            # Large diff => remove_left_padding/recover_left_padding/Float16Module/fbf machinery is the cause.
            if os.environ.get("SKYRL_ZEROKL_FWD_PROBE") == "1":
                try:
                    mb0 = micro_batches[0]
                    _seq = mb0["sequences"][:1]
                    _am = mb0["attention_mask"][:1].to(bool)
                    _na = int(mb0["num_actions"])
                    _rseq = _seq[0][_am[0]].unsqueeze(0)
                    _L = _rseq.shape[1]
                    _pos = torch.arange(_L, device=_rseq.device).unsqueeze(0)
                    _inner = self.actor_module[0]
                    for _ in range(4):
                        _inner = _inner.module if hasattr(_inner, "module") else _inner
                    with torch.no_grad(), _zerokl_scoring_ctx():
                        _dl = _inner(input_ids=_rseq, position_ids=_pos, attention_mask=None)[0].float()
                    _dlp = torch.log_softmax(_dl, dim=-1)
                    _rids = _rseq[0]
                    _dresp = _dlp[_L - _na - 1:_L - 1].gather(-1, _rids[_L - _na:].unsqueeze(-1)).squeeze(-1)
                    _d = (_dresp - log_probs[0, -_na:].float()).abs()
                    print(f"[ZEROKL-FWDPROBE] direct(bare GPTModel, unpadded) vs forward_backward_func: "
                          f"max={float(_d.max()):.3e} mean={float(_d.mean()):.3e} L={_L} na={_na}", flush=True)
                    # mp-worker prints don't surface in the run log -> also dump to shared storage.
                    try:
                        # which token positions diverge (relative to the na response tokens)
                        _bad = (_d > 1e-4).nonzero(as_tuple=True)[0].tolist()[:12]
                        with open("/mnt/local_storage/zerokl_probe.log", "a") as _pf:
                            _pf.write(f"FWDPROBE max={float(_d.max()):.3e} mean={float(_d.mean()):.3e} "
                                      f"L={_L} na={_na} bad_resp_idx={_bad} "
                                      f"vals={[round(float(_d[i]),4) for i in _bad]}\n")
                    except Exception:
                        pass
                except Exception as _e:
                    print(f"[ZEROKL-FWDPROBE] failed: {type(_e).__name__}: {_e}", flush=True)
                    try:
                        with open("/mnt/local_storage/zerokl_probe.log", "a") as _pf:
                            _pf.write(f"FWDPROBE FAILED: {type(_e).__name__}: {_e}\n")
                    except Exception:
                        pass
        else:
            # return dummy tensor for non-last pp stages
            device = micro_batches[0]["sequences"].device
            log_probs = torch.zeros(size=(1, 1), dtype=torch.bfloat16, device=device)
        return log_probs

    def forward_backward_mini_batch(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
        forward_only: bool = False,
    ) -> List[dict]:
        """
        Run forward-backward over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: A list of micro-batch dicts. Each dict must contain keys:
                "sequences", "attention_mask", "position_ids", "num_actions",
                "old_action_log_probs", "base_action_log_probs", "advantages",
                "loss_mask", "rollout_action_logprobs".
            seq_len: Sequence length (tokens) per sample (assumed same across micros after padding).
            micro_batch_size: Micro-batch size per forward pass.
            temperature: Optional temperature for logits scaling.
            loss_fn: Optional loss function name (e.g., "cross_entropy", "ppo").
                     If provided, overrides the config's policy_loss_type.
            loss_fn_config: Optional config overrides for the loss function.
            forward_only: If True, run the forward pass without backward (no gradients).
                          Useful for evaluation / loss-only inference paths (e.g., SFT
                          ``forward(loss_fn=...)`` codepath).

        Returns:
            List[dict]: one metrics dict per micro-batch in order.
        """
        forward_backward_func = get_forward_backward_func()

        # Resolve loss function
        resolved_loss_name = loss_fn if loss_fn is not None else self.cfg.algorithm.policy_loss_type
        if loss_fn is not None:
            current_loss_fn = PolicyLossRegistry.get(loss_fn)
        else:
            current_loss_fn = self.policy_loss_fn

        # Build config for loss function, applying any overrides
        loss_config = self.cfg.algorithm
        if loss_fn_config is not None:

            new_loss_config = OmegaConf.merge(OmegaConf.create(asdict(loss_config)), OmegaConf.create(loss_fn_config))
            # NOTE: users can provide a custom loss config class, so we need to use the same class after applying overrides
            loss_config = type(loss_config).from_dict_config(new_loss_config)

        def loss_func(logits, data):
            sequences = data["sequences"]
            packed_seq_params = data.get("packed_seq_params")
            packed_targets = data.get("packed_targets")
            num_actions = data["num_actions"]
            old_action_log_probs = data["old_action_log_probs"]
            base_action_log_probs = data["base_action_log_probs"]
            advantages = data["advantages"]
            loss_mask = data["loss_mask"]
            rollout_action_logprobs = data["rollout_action_logprobs"]
            action_mask = data.get("action_mask")
            num_microbatches = data.get("num_microbatches")
            # Number of microbatches carrying real samples (excludes fully-padding
            # microbatches added by token-based batching). Used to normalize the
            # KL/entropy terms over real microbatches only. Falls back to
            # num_microbatches when not provided (no padding microbatches).
            num_real_microbatches = data.get("num_real_microbatches", num_microbatches)

            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # temperature normalization
            if temperature != 1.0:
                logits.div_(temperature)

            if packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    group=tp_grp,
                    inference_only=False,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    sequences,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    tp_group=tp_grp,
                    inference_only=False,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
                )

            action_log_probs = token_logprobs[:, -num_actions:]

            # SkyRL-ZeroKL KLTRACE: print max|trainer - rollout| for the FIRST several micro-batches
            # UNCONDITIONALLY (no >0.05 filter). If micro-batch #0 (before any optimizer step) is ~0 and
            # later ones grow, the live residual is within-batch optimizer DRIFT, not a forward mismatch
            # (the layer-wise trace proved the forward is bitwise given identical weights).
            _zk_r0k = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
            if (os.environ.get("SKYRL_ZEROKL_SEQ_PROBE") == "1" and _zk_r0k
                    and rollout_action_logprobs is not None):
                _c = getattr(self, "_zk_kltrace_n", 0)
                if _c < 16:
                    self._zk_kltrace_n = _c + 1
                    _dm = (action_log_probs.detach().float() - rollout_action_logprobs.detach().float()).abs()
                    _msg = (f"[ZK-KLTRACE] mb#{_c} max|trn-rol|={float(_dm.max()):.4f} "
                            f"mean={float(_dm.mean()):.5f} n={_dm.numel()}")
                    print(_msg, flush=True)
                    try:
                        with open("/mnt/local_storage/zerokl_kltrace.log", "a") as _kf:
                            _kf.write(_msg + "\n")
                    except Exception:
                        pass

            # SkyRL-ZeroKL SEQ PROBE: dump the EXACT tokens the trainer scored + both logprob streams
            # (trainer action_log_probs vs engine rollout_action_logprobs) for the first real micro-batch,
            # so the residual can be re-scored offline through the engine on these very tokens. This
            # separates the three remaining hypotheses: (a) the saved tokens are NOT the engine's
            # trajectory -> generator assembly/alignment bug; (b) engine-prefill(saved) == trainer but !=
            # rollout -> rollout logprob source; (c) all equal -> residual is a forward-input (positions/
            # padding). Gated + rank-0 + once. Skipped for packed path (targets differ).
            _zk_r0p = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
            if (os.environ.get("SKYRL_ZEROKL_SEQ_PROBE") == "1" and _zk_r0p
                    and packed_seq_params is None and not getattr(self, "_zk_seq_probe_done", False)):
                try:
                    _rl = rollout_action_logprobs
                    _absd = (action_log_probs.detach().float() - _rl.detach().float()).abs() if _rl is not None else None
                    # only bother if this micro-batch actually carries the residual (an outlier present)
                    if _absd is not None and float(_absd.max()) > 0.05:
                        self._zk_seq_probe_done = True
                        torch.save({
                            "sequences": sequences.detach().cpu(),                    # [B, L] full padded ctx
                            "attention_mask": data["attention_mask"].detach().cpu(),  # locate real tokens
                            "position_ids": data.get("position_ids").detach().cpu()
                            if data.get("position_ids") is not None else None,
                            "num_actions": int(num_actions),
                            "action_log_probs": action_log_probs.detach().cpu(),      # trainer [B, na]
                            "rollout_action_logprobs": _rl.detach().cpu(),            # engine  [B, na]
                            "loss_mask": loss_mask.detach().cpu() if loss_mask is not None else None,
                        }, "/mnt/local_storage/zerokl_seq_probe.pt")
                        print(f"[ZEROKL-SEQPROBE] saved 1 micro-batch (B={sequences.shape[0]} L={sequences.shape[1]} "
                              f"na={int(num_actions)} max|trn-rol|={float(_absd.max()):.3f}) -> zerokl_seq_probe.pt",
                              flush=True)
                except Exception as _e:
                    print(f"[ZEROKL-SEQPROBE] failed: {type(_e).__name__}: {_e}", flush=True)

            # policy loss should be calculated based on the selected token logprobs
            policy_loss, loss_metrics = current_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=loss_config,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )

            # SFT path: cross_entropy loss (negative log likelihood)
            if resolved_loss_name == "cross_entropy":
                loss = policy_loss

                # Compute elementwise loss for Tinker API (per-token NLL)
                with torch.no_grad():
                    elementwise_loss = -action_log_probs
                    if loss_mask is not None:
                        elementwise_loss = elementwise_loss * loss_mask

                # Build per-sequence loss_fn_outputs.
                # Compute valid_lens vectorized on GPU, then move tensors to CPU
                # exactly once before iterating in Python — avoids ~3N GPU->CPU
                # syncs per micro-batch (item()/cpu()/tolist() inside the loop).
                batch_size = action_log_probs.shape[0]
                seq_len = action_log_probs.shape[1]
                if action_mask is not None:
                    valid_lens_t = action_mask.sum(dim=-1).long()
                elif loss_mask is not None:
                    valid_lens_t = loss_mask.sum(dim=-1).long()
                else:
                    valid_lens_t = torch.full((batch_size,), seq_len, device=action_log_probs.device, dtype=torch.long)

                # Bulk GPU->CPU sync: one transfer for logprobs, elementwise_loss, and valid_lens.
                action_log_probs_cpu = action_log_probs.detach().cpu()
                elementwise_loss_cpu = elementwise_loss.detach().cpu()
                valid_lens = valid_lens_t.cpu().tolist()

                loss_fn_outputs = []
                for i in range(batch_size):
                    valid_len = valid_lens[i]
                    loss_fn_outputs.append(
                        {
                            "logprobs": (action_log_probs_cpu[i, -valid_len:].tolist() if valid_len > 0 else []),
                            "elementwise_loss": (
                                elementwise_loss_cpu[i, -valid_len:].tolist() if valid_len > 0 else []
                            ),
                        }
                    )

                metrics = {
                    "loss": loss.item(),
                    "response_length": num_actions,
                    "loss_fn_outputs": loss_fn_outputs,
                }
                return loss, metrics

            # RL path: add optional KL/entropy terms
            with torch.set_grad_enabled(loss_config.use_entropy_loss):
                if packed_seq_params is not None and packed_targets is not None:
                    entropy, entropy_for_loss = vocab_parallel_entropy_packed_sequences(
                        logits,
                        packed_seq_params.cu_seqlens_q_padded,
                        sequences.shape[1],
                        num_actions,
                        data["attention_mask"],
                        loss_mask,
                        mpu.get_context_parallel_group(),
                        sub_seq_lengths=data.get("sub_seq_lengths_list"),
                    )
                else:
                    action_logits = logits[:, -num_actions - 1 : -1, :]
                    entropy_BS = vocab_parallel_entropy(action_logits)
                    entropy = masked_mean(entropy_BS, loss_mask)
                    entropy_for_loss = entropy

            if loss_config.use_entropy_loss:
                entropy_loss_term = entropy_for_loss * loss_config.entropy_loss_coef
            else:
                entropy_loss_term = torch.tensor(0.0, device=logits.device)

            if loss_config.use_kl_loss:
                kl_loss = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    loss_mask=loss_mask,
                    kl_estimator_type=loss_config.kl_estimator_type,
                )
                kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
            else:
                kl_loss = torch.tensor(0.0, device=logits.device)
            kl_loss_term = kl_loss * loss_config.kl_loss_coef

            # Policy losses are pre-scaled to achieve the correct loss_reduction
            # when summing across the entire minibatch (see `apply_loss_reduction_to_advantages_minibatch`).
            # Megatron divides loss by num_microbatches
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/pipeline_parallel/schedules.py#L248)
            # and the data parallel all-reduce averages gradients across dp_size.
            # Megatron's schedule separately multiplies loss by the CP size for two-output loss funcs,
            # so CP ranks are not included in this correction factor.
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/distributed/distributed_data_parallel.py#L285)
            # so we multiply by both factors to recover the correct sum reduction.
            grad_sum_correction_factor = num_microbatches * dp_size

            # NOTE: The KL and entropy loss terms are not pre-scaled,
            # so we just average them across microbatches and DP workers.
            # KL and entropy use Megatron's existing microbatch and CP schedule scaling.
            # Megatron divides by num_microbatches (which includes fully-padding microbatches
            # added by token-based batching). Those padding microbatches contribute 0 to
            # KL/entropy, so dividing by the full count would dilute the regularization by
            # num_real/num_total. Scale up by num_microbatches/num_real_microbatches so the
            # terms are averaged over real microbatches only (no-op when there is no padding).
            kl_entropy_microbatch_scale = num_microbatches / max(1, num_real_microbatches)
            loss = (
                policy_loss * grad_sum_correction_factor
                + (kl_loss_term - entropy_loss_term) * kl_entropy_microbatch_scale
            )
            unscaled_loss = loss / grad_sum_correction_factor

            # Build per-sequence loss_fn_outputs with logprobs.
            batch_size = action_log_probs.shape[0]
            seq_len = action_log_probs.shape[1]

            if action_mask is not None:
                valid_lens = action_mask.sum(dim=1).int().tolist()
            elif loss_mask is not None:
                valid_lens = loss_mask.sum(dim=1).int().tolist()
            else:
                valid_lens = [seq_len] * batch_size

            detached_log_probs = action_log_probs.detach().cpu()
            loss_fn_outputs = []
            for i, valid_len in enumerate(valid_lens):
                loss_fn_outputs.append(
                    {
                        "logprobs": detached_log_probs[i, -valid_len:].tolist() if valid_len > 0 else [],
                    }
                )

            metrics = {
                "final_loss": unscaled_loss.detach().item(),
                "policy_loss": policy_loss.detach().item(),
                "policy_entropy": entropy.detach().item(),
                "policy_kl": kl_loss.detach().item(),
                "loss_fn_outputs": loss_fn_outputs,
            }
            for k, v in loss_metrics.items():
                metrics["loss_metrics/" + k] = v
            metrics.update(
                compute_minibatch_rollout_logprob_diff_metrics(action_log_probs, rollout_action_logprobs, loss_mask)
            )
            return loss, metrics

        def forward_step(batch_iter, model):
            # NOTE(Charlie): despite the name, methods like `remove_left_padding()` are padding-agnostic
            # (can be left, or right) as it uses attention_mask to locate real tokens. Same thing
            # for recover_left_padding and setup_per_microbatch_replay_forward. Especially relevant
            # after this PR https://github.com/NovaSky-AI/SkyRL/pull/1285.
            batch = next(batch_iter)

            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=get_model_config(model),
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            # When present, sub_seq_lengths enumerates every sub-sequence
            # inside every row of the micro-batch (controller-side mini-batch
            # packing). preprocess_packed_seqs uses it to emit cu_seqlens
            # entries covering all sub-seqs, not one per row.
            #
            # It arrives as a ``TensorList`` data field.
            # ``preprocess_packed_seqs`` and the packed-logprob scatter use
            # ``list[list[int]]``, so convert tensors -> python lists here.
            sub_seq_lengths_field = batch.get("sub_seq_lengths")
            sub_seq_lengths = [t.tolist() for t in sub_seq_lengths_field] if sub_seq_lengths_field is not None else None
            batch["sub_seq_lengths_list"] = sub_seq_lengths

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                    sub_seq_lengths=sub_seq_lengths,
                )
                batch["packed_seq_params"] = packed_seq_params
                batch["packed_targets"] = _build_packed_targets(
                    sequences, attention_mask, packed_seq_params, sub_seq_lengths=sub_seq_lengths
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            # SkyRL-ZeroKL: pass attention_mask=None (pure causal flash) to match the engine's
            # causal-flash path. new_sequences is unpadded + decoder-causal; an explicit mask sends
            # TE down a different flash variant -> diffuse ~0.01 logprob drift vs the engine.
            _zk_mask = None if (os.environ.get("SKYRL_ZERO_KL") == "1" and packed_seq_params is None) else new_attention_mask
            outputs = model(
                new_sequences,
                new_position_ids,
                _zk_mask,
                packed_seq_params=packed_seq_params,
            )

            if not self.remove_microbatch_padding:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_backward()

            return outputs, partial(loss_func, data=batch)

        # batch should be a list of micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # SkyRL-ZeroKL bisect: checksum the weights the TRAINER FORWARD actually reads, in the SAME
        # formula as native_weight_sync.extract_native_weights (the SENDER) and gptmodel_vllm
        # (the ENGINE runtime probe). Three-way compare localizes the multi-process 0.0104:
        #   trainer-fwd == SENDER  but  ENGINE != SENDER   -> sync/cumem delivers wrong weights
        #   trainer-fwd == ENGINE  == SENDER               -> weights fine; diff is token alignment
        # rank-0 only: all DP replicas hold identical weights, and 8 workers racing one "w" path
        # truncated the file to 0 bytes. Write from a single rank so the dump is complete + clean.
        _zk_is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        if (os.environ.get("SKYRL_ZEROKL_BISECT") == "1" and _zk_is_rank0
                and not getattr(self, "_zk_fwd_cksum_done", False)):
            from skyrl.backends.skyrl_train.zerokl.weight_fingerprint import fingerprint_line
            _wf = None
            try:
                _wf = open("/mnt/local_storage/zerokl_trn_w.txt", "w")
            except Exception:
                _wf = None
            with torch.no_grad():
                _s, _n, _seen = 0.0, 0, set()
                for _m in self.actor_module:
                    _inner = _m
                    for _ in range(4):
                        _inner = _inner.module if hasattr(_inner, "module") else _inner
                    for _nm, _p in _inner.named_parameters():
                        if _nm in _seen or _nm.startswith("mtp."):
                            continue
                        _seen.add(_nm)
                        _t = _p.detach()
                        if hasattr(_t, "full_tensor"):
                            try:
                                _t = _t.full_tensor()
                            except Exception:
                                pass
                        # 3-way element-wise fingerprint (== engine dump) so compare_weight_fingerprints
                        # can prove the sync is byte-identical, not just abs-sum-equal.
                        if _wf is not None:
                            try:
                                _wf.write(fingerprint_line(_nm, _t))
                            except Exception:
                                pass
                        _s += float(_t.to(torch.bfloat16).float().double().abs().sum()); _n += 1
            if _wf is not None:
                try:
                    _wf.close()
                    print("[ZEROKL-WDUMP] trainer weights -> zerokl_trn_w.txt", flush=True)
                except Exception:
                    pass
            print(f"[ZEROKL-BISECT] TRAINER forward-weight non-MTP cksum={_s:.6f} (n={_n})  "
                  f"[compare to SENDER + ENGINE]", flush=True)
            try:
                with open("/mnt/local_storage/zerokl_probe.log", "a") as _pf:
                    _pf.write(f"BISECT-TRAINER cksum={_s:.6f} n={_n}\n")
            except Exception:
                pass
            self._zk_fwd_cksum_done = True

        # SkyRL-ZeroKL LAYERTRACE: dump EVERY submodule output of the ACTUAL fbf/Float16Module training
        # forward (micro-batch 0, rank-0, once) + that micro-batch's tokens, so the distributed-machinery
        # divergence can be localized against a clean single-process forward (trace_layerwise.py). The bare
        # GPTModel submodules ARE the objects Float16Module wraps, so hooks fire during the real forward.
        _zk_lt_handles = []
        if (os.environ.get("SKYRL_ZEROKL_SEQ_PROBE") == "1" and _zk_is_rank0
                and not getattr(self, "_zk_ltrace_done", False)):
            self._zk_ltrace_done = True
            self._zk_lt_store = {}
            _lt = self._zk_lt_store
            _inner_lt = self.actor_module[0]
            for _ in range(4):
                _inner_lt = _inner_lt.module if hasattr(_inner_lt, "module") else _inner_lt

            def _mk_lt(_name):
                def _h(_m, _i, _o):
                    _t = _o[0] if isinstance(_o, tuple) else _o
                    if torch.is_tensor(_t) and _name not in _lt:
                        _lt[_name] = _t.detach().float().cpu()
                return _h

            def _mk_lt_in(_name):
                # capture the INPUTS (post-RoPE q/k/v) to core_attention so we can tell whether the
                # divergence is the kernel or upstream (RoPE/split). key = _name + "::IN<idx>".
                def _hpre(_m, _args):
                    for _j, _a in enumerate(_args):
                        _k = f"{_name}::IN{_j}"
                        if torch.is_tensor(_a) and _k not in _lt:
                            _lt[_k] = _a.detach().float().cpu()
                return _hpre
            for _nm, _mod in _inner_lt.named_modules():
                if _nm:
                    _zk_lt_handles.append(_mod.register_forward_hook(_mk_lt(_nm)))
                    if _nm.endswith("core_attention"):
                        _zk_lt_handles.append(_mod.register_forward_pre_hook(_mk_lt_in(_nm)))
            try:
                _mb0 = micro_batches[0]
                torch.save({"sequences": _mb0["sequences"].detach().cpu(),
                            "attention_mask": _mb0["attention_mask"].detach().cpu(),
                            "num_actions": int(_mb0["num_actions"]),
                            "rollout_action_logprobs": _mb0["rollout_action_logprobs"].detach().cpu()
                            if _mb0.get("rollout_action_logprobs") is not None else None,
                            "position_ids": _mb0.get("position_ids").detach().cpu()
                            if _mb0.get("position_ids") is not None else None},
                           "/mnt/local_storage/trace_live_tokens.pt")
            except Exception as _e:
                print(f"[ZEROKL-LAYERTRACE] token save failed: {_e}", flush=True)

        with _zerokl_scoring_ctx():
            metrics_list = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=len(micro_batches),
                seq_length=seq_len,
                micro_batch_size=micro_batch_size,
                forward_only=forward_only,
            )

        if _zk_lt_handles:
            for _h in _zk_lt_handles:
                _h.remove()
            try:
                torch.save(self._zk_lt_store, "/mnt/local_storage/trace_live_trainer.pt")
                print(f"[ZEROKL-LAYERTRACE] dumped {len(self._zk_lt_store)} live-trainer intermediates "
                      f"-> trace_live_trainer.pt", flush=True)
            except Exception as _e:
                print(f"[ZEROKL-LAYERTRACE] dump failed: {_e}", flush=True)

        # broadcast metrics to all pp ranks
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            metrics_list = [None] * len(micro_batches)
        with torch.no_grad():
            torch.distributed.broadcast_object_list(
                metrics_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        return metrics_list
