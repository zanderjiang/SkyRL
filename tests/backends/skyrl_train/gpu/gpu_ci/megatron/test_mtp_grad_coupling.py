"""Decisive DISTRIBUTED coupling probe (TP=4, DP=2 -- the failing config).

The existing isolation probes run at TP=1/DP=1, where Megatron's distributed
optimizer is trivial (no grad-buffer sharding / reduce-scatter) and TP has no
all-reduce. They show the *forward* is decoupled. But the entropy collapse only
appears at TP=4, DP=2, so any coupling must live in the distributed grad path
(TP all-reduce of the shared output/embedding backward, or the distributed
optimizer) that those probes never exercise.

This probe runs the REAL training forward_backward (the exact code path) on a
FIXED batch at two draft-loss weights and asks one question:

    Does the POLICY parameters' gradient change when the draft loss is added?

If the design is truly decoupled, the policy main_grad at draft weight 0.0 must
equal the policy main_grad at weight 0.5 (the draft loss only touches .mtp.
params). Any difference is the leak that reshapes the policy.

Also verifies the prior investigation's concrete anomalies:
  - does ``zero_grad_buffer()`` actually zero the .mtp. params' main_grad?
  - do .mtp. and policy params alias the same grad-buffer storage?

Run::
    NVTE_FLASH_ATTN=0 uv run --isolated --extra megatron --extra dev pytest -s -vvv \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_mtp_grad_coupling.py
"""

import pytest
import ray

from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as _megatron_worker_mod,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    make_dummy_training_batch,
)

MODEL_NAME = "XiaomiMiMo/MiMo-7B-RL"


class _CouplingWorker(MegatronPolicyWorkerBase):
    def set_mtp_weight(self, w: float):
        self.cfg.policy.megatron_config.mtp_loss_weight = float(w)
        return float(w)

    def probe_cfull_isolation(self):
        """Verify C-full actually isolates the MTP head: the policy DDP grad buffer must contain NO
        ``.mtp`` params (so the policy reduction is byte-identical to a no-MTP model), and every
        ``.mtp`` param must live in the separate MTP DDP buffer instead."""
        from megatron.core.utils import unwrap_model

        def _buffer_param_ids(ddp):
            ids = set()
            for attr in ("buffers", "grad_buffers"):
                for buf in getattr(ddp, attr, []) or []:
                    for p in getattr(buf, "params", []) or []:
                        ids.add(id(p))
            return ids

        gm = unwrap_model(self.actor_module[0])
        policy_buf_ids = _buffer_param_ids(self.actor_module[0])
        mtp_buf_ids = _buffer_param_ids(self._mtp_separate.mtp_ddp) if self._mtp_separate is not None else set()

        mtp_total = mtp_in_policy_buf = mtp_in_mtp_buf = mtp_requires_grad = 0
        policy_in_policy_buf = policy_total = 0
        for name, p in gm.named_parameters():
            if ".mtp." in name or name.startswith("mtp."):
                mtp_total += 1
                mtp_requires_grad += int(p.requires_grad)
                mtp_in_policy_buf += int(id(p) in policy_buf_ids)
                mtp_in_mtp_buf += int(id(p) in mtp_buf_ids)
            else:
                policy_total += 1
                policy_in_policy_buf += int(id(p) in policy_buf_ids)
        return {
            "cfull_enabled": bool(getattr(self, "_mtp_cfull_enabled", False)),
            "mtp_separate_built": self._mtp_separate is not None,
            "mtp_total": mtp_total,
            "mtp_requires_grad": mtp_requires_grad,
            "mtp_in_POLICY_buffer": mtp_in_policy_buf,  # MUST be 0
            "mtp_in_MTP_buffer": mtp_in_mtp_buf,  # should equal mtp_total
            "policy_total": policy_total,
            "policy_in_POLICY_buffer": policy_in_policy_buf,
            "policy_buffer_size": len(policy_buf_ids),
        }

    def snapshot_policy_grad(self, tag: str):
        """Clone every policy (non-mtp) param's main_grad and record group norms."""
        from megatron.core.utils import unwrap_model

        if not hasattr(self, "_snaps"):
            self._snaps = {}
        gm = unwrap_model(self.actor_module[0])
        grads = {}
        mtp_norm_sq = 0.0
        policy_norm_sq = 0.0
        for name, p in gm.named_parameters():
            mg = getattr(p, "main_grad", None)
            if mg is None:
                continue
            n2 = float(mg.detach().float().pow(2).sum().item())
            if ".mtp." in name or name.startswith("mtp."):
                mtp_norm_sq += n2
            else:
                policy_norm_sq += n2
                grads[name] = mg.detach().float().clone()
        self._snaps[tag] = grads
        return {"tag": tag, "policy_grad_norm": policy_norm_sq**0.5, "mtp_grad_norm": mtp_norm_sq**0.5}

    def compare_snapshots(self, tag_a: str, tag_b: str):
        a = self._snaps[tag_a]
        b = self._snaps[tag_b]
        max_abs = 0.0
        sum_sq_diff = 0.0
        sum_sq_a = 0.0
        worst = None
        for name, ga in a.items():
            gb = b.get(name)
            if gb is None:
                continue
            diff = gb - ga
            m = float(diff.abs().max().item())
            sum_sq_diff += float(diff.pow(2).sum().item())
            sum_sq_a += float(ga.pow(2).sum().item())
            if m > max_abs:
                max_abs = m
                worst = name
        return {
            "max_abs_diff": max_abs,
            "worst_param": worst,
            "rel_l2_diff": (sum_sq_diff**0.5) / (sum_sq_a**0.5 + 1e-12),
            "num_params": len(a),
        }

    def probe_draft_autograd_leak(self, seq_len: int = 96, topk: int = 256) -> dict:
        """Backprop ONLY the draft loss (the real soft-CE topk path) and use autograd.grad to ask:
        does it reach the policy backbone? Run at TP>1 to expose any SP/TP-specific leak the TP=1
        isolation probe misses. Also separately checks whether the native in-forward MTP loss is
        active by autograd-ing the MAIN logits (policy path) against the mtp params."""
        import torch
        from megatron.core import parallel_state as mpu
        from megatron.core.utils import unwrap_model

        from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
        from skyrl.backends.skyrl_train.mtp.hidden_capture import (
            MTPHiddenCapture,
            _resolve_mtp_host,
            _unwrap_model,
        )
        from skyrl.backends.skyrl_train.mtp.soft_ce import (
            draft_soft_ce_topk,
            shift_mask_for_mtp,
        )

        gm = unwrap_model(self.actor_module[0])
        host = _resolve_mtp_host(_unwrap_model(self.actor_module[0]))
        device = torch.cuda.current_device()
        ids = (torch.arange(seq_len, device=device) * 7 + 3) % 100000
        sequences = ids.unsqueeze(0)
        attention_mask = torch.ones_like(sequences)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        tp_grp = mpu.get_tensor_model_parallel_group()

        def to_bsv(x):
            if x.dim() == 3 and x.shape[0] != 1 and x.shape[1] == 1:
                return x.transpose(0, 1).contiguous()
            return x

        # Pick a representative set of policy params to test for leak.
        def pick(nm):
            return (".mtp." not in nm and not nm.startswith("mtp.")) and (
                "embedding" in nm
                or "output_layer" in nm
                or "layers.0." in nm
                or "layers.1." in nm
                or "final_layernorm" in nm
            )

        policy_named = [(n, p) for n, p in gm.named_parameters() if pick(n) and p.requires_grad]
        policy_params = [p for _, p in policy_named]

        capture = MTPHiddenCapture(self.actor_module[0], detach_trunk=True, detach_shared_embedding=True)
        with capture.capture():
            outputs = self.actor_module[0](sequences, position_ids, attention_mask)
            student_hidden = capture.compute_student_hidden_states()
        student_logits = to_bsv(project_mtp_hidden_to_logits(student_hidden, host, detach_output_weight=True)[0])
        teacher_logits = to_bsv(outputs)
        mask = torch.ones(student_logits.shape[0], student_logits.shape[1], device=device)
        layer_mask = shift_mask_for_mtp(mask, 0)
        draft_loss = draft_soft_ce_topk(
            student_logits, teacher_logits, layer_mask, k=topk, vocab_parallel_group=tp_grp, roll_shift=1
        )

        # (1) Does the DRAFT loss reach the policy backbone via autograd?
        grads = torch.autograd.grad(draft_loss, policy_params, allow_unused=True, retain_graph=False)
        draft_leak = [
            (policy_named[i][0], float(g.detach().float().norm().item()))
            for i, g in enumerate(grads)
            if g is not None and float(g.detach().float().norm().item()) > 0
        ]
        draft_leak.sort(key=lambda kv: kv[1], reverse=True)

        # (2) Is the NATIVE in-forward MTP loss active? Check if the MAIN logits depend on mtp params.
        out2 = self.actor_module[0](sequences, position_ids, attention_mask)
        main = to_bsv(out2).float()
        pce = -torch.log_softmax(main[:, :-1], dim=-1).gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1).mean()
        mtp_named = [
            (n, p) for n, p in gm.named_parameters() if (".mtp." in n or n.startswith("mtp.")) and p.requires_grad
        ]
        mtp_params = [p for _, p in mtp_named]
        mgrads = torch.autograd.grad(pce, mtp_params, allow_unused=True, retain_graph=False)
        native_into_mtp = [
            (mtp_named[i][0], float(g.detach().float().norm().item()))
            for i, g in enumerate(mgrads)
            if g is not None and float(g.detach().float().norm().item()) > 0
        ]

        return {
            "tp_size": mpu.get_tensor_model_parallel_world_size(),
            "draft_loss": float(draft_loss.detach().item()),
            "num_policy_params_tested": len(policy_params),
            "DRAFT_leaks_to_policy_via_autograd": draft_leak[:12],
            "main_logits_depend_on_mtp(native_active)": native_into_mtp[:12],
        }

    def probe_packed_draft_autograd_leak(self, topk: int = 256) -> dict:
        """Run the PACKED draft path (remove_microbatch_padding=True, the real failing config) at this
        TP and backprop ONLY the draft loss, then check main_grad on every policy param. A raw
        backward (not through forward_backward_func) does NOT trigger finalize/reduce-scatter, so
        main_grad is the pure per-rank autograd grad -- no RS confound."""
        import torch
        from megatron.core import parallel_state as mpu
        from megatron.core.utils import unwrap_model

        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )
        from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
        from skyrl.backends.skyrl_train.mtp.hidden_capture import (
            MTPHiddenCapture,
            _resolve_mtp_host,
            _unwrap_model,
        )
        from skyrl.backends.skyrl_train.mtp.soft_ce import (
            draft_soft_ce_topk,
            shift_mask_for_mtp,
        )
        from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
            _build_packed_valid_mask,
        )

        gm = unwrap_model(self.actor_module[0])
        host = _resolve_mtp_host(_unwrap_model(self.actor_module[0]))
        device = torch.cuda.current_device()
        ids = list(range(3, 99))
        a, b = ids[:60], ids[:40]
        S, pad_id = 60, 0

        def leftpad(x):
            return [pad_id] * (S - len(x)) + x

        sequences = torch.tensor([leftpad(a), leftpad(b)], device=device, dtype=torch.long)
        attention_mask = torch.tensor(
            [[0] * (S - len(a)) + [1] * len(a), [0] * (S - len(b)) + [1] * len(b)],
            device=device,
            dtype=torch.bool,
        )
        new_sequences, packed_seq_params = preprocess_packed_seqs(
            sequences, attention_mask, pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True)
        )
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        new_position_ids = preprocess_packed_seqs(
            position_ids, attention_mask, pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True)
        )[0]

        capture = MTPHiddenCapture(self.actor_module[0], detach_trunk=True, detach_shared_embedding=True)
        with capture.capture():
            outputs = self.actor_module[0](new_sequences, new_position_ids, None, packed_seq_params=packed_seq_params)
            student_hidden = capture.compute_student_hidden_states()

        def to_bsv(x):
            if x.dim() == 3 and x.shape[0] != 1 and x.shape[1] == 1:
                return x.transpose(0, 1).contiguous()
            return x

        student_logits = to_bsv(project_mtp_hidden_to_logits(student_hidden, host, detach_output_weight=True)[0])
        teacher_logits = to_bsv(outputs)
        packed_mask = _build_packed_valid_mask(attention_mask, packed_seq_params).to(device)
        cu = packed_seq_params.cu_seqlens_q_padded
        layer_mask = shift_mask_for_mtp(packed_mask, 0, cu_seqlens=cu)
        tp_grp = mpu.get_tensor_model_parallel_group()
        draft_loss = draft_soft_ce_topk(
            student_logits, teacher_logits, layer_mask, k=topk, vocab_parallel_group=tp_grp, roll_shift=1
        )

        for chunk in self.actor_module:
            try:
                chunk.zero_grad_buffer()
            except Exception:
                pass
        self.actor_module[0].zero_grad(set_to_none=True)
        draft_loss.backward()

        def gradnorm(p):
            mg = getattr(p, "main_grad", None)
            g = mg if mg is not None else p.grad
            return float(g.detach().float().norm().item()) if g is not None else 0.0

        leak = []
        mtp_with_grad = 0
        for name, p in gm.named_parameters():
            is_mtp = ".mtp." in name or name.startswith("mtp.")
            gn = gradnorm(p)
            if is_mtp and gn > 0:
                mtp_with_grad += 1
            elif not is_mtp and gn > 1e-9:
                leak.append((name, gn))
        leak.sort(key=lambda kv: kv[1], reverse=True)
        return {
            "tp_size": mpu.get_tensor_model_parallel_world_size(),
            "path": "PACKED",
            "draft_loss": float(draft_loss.detach().item()),
            "mtp_params_with_grad": mtp_with_grad,
            "num_policy_params_leaking": len(leak),
            "LEAK_top12": leak[:12],
        }

    def probe_mtp_grad_breakdown(self, seq_len: int = 96, topk: int = 256) -> dict:
        """Backprop the real draft loss and report per-mtp-PARAM grad norms (sorted), plus the
        output_weight / embedding-weight norms. Pinpoints WHICH mtp param carries the huge gradient
        that dominates the global grad-norm on MiMo (vs Qwen3.5 where it's tiny)."""
        import torch
        from megatron.core import parallel_state as mpu
        from megatron.core.utils import unwrap_model

        from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
        from skyrl.backends.skyrl_train.mtp.hidden_capture import (
            MTPHiddenCapture,
            _resolve_mtp_host,
            _unwrap_model,
        )
        from skyrl.backends.skyrl_train.mtp.soft_ce import (
            draft_soft_ce_topk,
            shift_mask_for_mtp,
        )

        gm = unwrap_model(self.actor_module[0])
        host = _resolve_mtp_host(_unwrap_model(self.actor_module[0]))
        device = torch.cuda.current_device()
        ids = (torch.arange(seq_len, device=device) * 7 + 3) % 100000
        sequences = ids.unsqueeze(0)
        attention_mask = torch.ones_like(sequences)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        tp_grp = mpu.get_tensor_model_parallel_group()

        def to_bsv(x):
            if x.dim() == 3 and x.shape[0] != 1 and x.shape[1] == 1:
                return x.transpose(0, 1).contiguous()
            return x

        capture = MTPHiddenCapture(self.actor_module[0], detach_trunk=True, detach_shared_embedding=True)
        with capture.capture():
            outputs = self.actor_module[0](sequences, position_ids, attention_mask)
            student_hidden = capture.compute_student_hidden_states()
        student_logits = to_bsv(project_mtp_hidden_to_logits(student_hidden, host, detach_output_weight=True)[0])
        teacher_logits = to_bsv(outputs)
        mask = torch.ones(student_logits.shape[0], student_logits.shape[1], device=device)
        layer_mask = shift_mask_for_mtp(mask, 0)
        draft_loss = draft_soft_ce_topk(
            student_logits, teacher_logits, layer_mask, k=topk, vocab_parallel_group=tp_grp, roll_shift=1
        )
        for chunk in self.actor_module:
            try:
                chunk.zero_grad_buffer()
            except Exception:
                pass
        self.actor_module[0].zero_grad(set_to_none=True)
        draft_loss.backward()

        per = []
        tot = 0.0
        for name, p in gm.named_parameters():
            if ".mtp." in name or name.startswith("mtp."):
                mg = getattr(p, "main_grad", None)
                g = mg if mg is not None else p.grad
                gn = float(g.detach().float().norm().item()) if g is not None else 0.0
                wn = float(p.detach().float().norm().item())
                per.append((name, round(gn, 4), round(wn, 4)))
                tot += gn * gn
        per.sort(key=lambda kv: kv[1], reverse=True)
        # output & embedding weight norms (the detached weights used in the draft projection)
        ow = getattr(getattr(host, "output_layer", None), "weight", None)
        ew = None
        emb = getattr(host, "embedding", None)
        if emb is not None:
            ew = getattr(getattr(emb, "word_embeddings", None), "weight", None)
        return {
            "draft_loss": round(float(draft_loss.item()), 4),
            "mtp_grad_global_norm": round(tot**0.5, 4),
            "per_mtp_param (name, grad_norm, weight_norm)": per,
            "output_layer.weight norm": round(float(ow.detach().float().norm().item()), 4) if ow is not None else None,
            "embedding.weight norm": round(float(ew.detach().float().norm().item()), 4) if ew is not None else None,
        }

    def policy_param_fingerprint(self) -> dict:
        """Global + per-tensor fingerprint of the POLICY (non-mtp) parameters (current values)."""
        from megatron.core.utils import unwrap_model

        gm = unwrap_model(self.actor_module[0])
        gsum = 0.0
        gsumsq = 0.0
        n = 0
        worst = {}
        for name, p in gm.named_parameters():
            if ".mtp." in name or name.startswith("mtp."):
                continue
            t = p.detach().double()
            gsum += float(t.sum().item())
            gsumsq += float((t * t).sum().item())
            n += 1
        return {"num_policy_params": n, "sum": round(gsum, 6), "sumsq": round(gsumsq, 6)}

    def probe_policy_logits_fingerprint(self, seq_len: int = 128) -> dict:
        """Run the policy forward on a FIXED deterministic input and return a full-precision
        fingerprint of the main logits. Compare MTP-on vs nosd: if the in-forward self.mtp(...) block
        (whose output is discarded via chunk[0]) perturbs the policy logits, they will differ. This is
        the last candidate channel after rollout/weights/gradient are all proven identical."""
        import torch

        device = torch.cuda.current_device()
        ids = (torch.arange(seq_len, device=device) * 7 + 3) % 100000
        sequences = ids.unsqueeze(0)
        attention_mask = torch.ones_like(sequences)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        self.model.eval() if hasattr(self, "model") else None
        with torch.no_grad():
            out = self.actor_module[0](sequences, position_ids, attention_mask)
        # out: [seq, batch, vocab/tp] or [batch, seq, vocab/tp]
        t = out.detach().float()
        # Reduce to a layout-independent fingerprint: per-position max-logit + a global checksum.
        flat = t.reshape(-1)
        return {
            "shape": list(t.shape),
            "sum": round(float(flat.double().sum().item()), 4),
            "sumsq": round(float((flat.double() ** 2).sum().item()), 4),
            "absmax": round(float(flat.abs().max().item()), 6),
            "first16": [round(float(x), 6) for x in flat[:16].tolist()],
        }

    def probe_export_fingerprint(self) -> dict:
        """Fingerprint every exported MAIN (non-mtp) HF tensor: name -> (shape, float64 sum, norm).
        Run for an MTP-enabled worker and a nosd worker (same checkpoint) and diff the fingerprints.
        If they differ, the MTP head's presence changes the main weights vLLM receives -> different
        rollout -> the collapse channel. (vLLM's MiMoModel.load_weights skips 'mtp_layers', so only a
        change in the MAIN tensors could alter the rollout.)"""
        from megatron.core import parallel_state as mpu

        fp = {}
        for name, tensor in self.bridge.export_hf_weights(self.actor_module, show_progress=False):
            low = name.lower()
            if "mtp" in low or "nextn" in low:
                continue
            t = tensor.detach().float()
            fp[name] = [list(tensor.shape), round(float(t.double().sum().item()), 6), round(float(t.norm().item()), 6)]
        return {
            "tp_size": mpu.get_tensor_model_parallel_world_size(),
            "num_main_tensors": len(fp),
            "fingerprint": fp,
        }

    def probe_export_coupling(self) -> dict:
        """Does training (drifting) the MTP head change the exported MAIN-model weights?

        The weight sync to vLLM is ``bridge.export_hf_weights(actor_module)``. The training worker's
        main params are provably identical with/without MTP, but if the EXPORT mixes the (drifting)
        mtp.* params into the main HF tensors (e.g. via shared embedding/output handling), vLLM would
        receive corrupted main weights -> off-policy rollouts -> entropy collapse, even with spec off.

        Test: export main HF tensors, perturb ONLY .mtp.* params (simulate drift), export again, and
        report any MAIN (non-mtp) HF tensor that changed."""
        import torch
        from megatron.core import parallel_state as mpu
        from megatron.core.utils import unwrap_model

        def export_main():
            out = {}
            for name, tensor in self.bridge.export_hf_weights(self.actor_module, show_progress=False):
                low = name.lower()
                if "mtp" in low or "nextn" in low:
                    continue
                out[name] = tensor.detach().float().clone().cpu()
            return out

        before = export_main()

        # Perturb only the mtp.* params (simulate the draft loss having trained/drifted them).
        gm = unwrap_model(self.actor_module[0])
        n_perturbed = 0
        with torch.no_grad():
            for name, p in gm.named_parameters():
                if ".mtp." in name or name.startswith("mtp."):
                    p.add_(torch.randn_like(p) * (p.std() + 1e-4) * 0.5)
                    n_perturbed += 1

        after = export_main()

        changed = []
        for name, t0 in before.items():
            t1 = after.get(name)
            if t1 is None:
                changed.append((name, "MISSING_AFTER"))
                continue
            if t0.shape != t1.shape:
                changed.append((name, f"shape {t0.shape}->{t1.shape}"))
                continue
            d = (t1 - t0).abs().max().item()
            if d > 0:
                changed.append((name, f"max_abs_change={d:.3e}"))
        return {
            "tp_size": mpu.get_tensor_model_parallel_world_size(),
            "num_mtp_params_perturbed": n_perturbed,
            "num_main_tensors_exported": len(before),
            "num_main_tensors_changed_by_mtp_perturbation": len(changed),
            "changed_examples": changed[:15],
        }

    def probe_buffer_layout(self):
        from megatron.core.utils import unwrap_model

        gm = unwrap_model(self.actor_module[0])
        for chunk in self.actor_module:
            chunk.zero_grad_buffer()

        mtp_nonzero = []
        mtp_has_main_grad = 0
        mtp_total = 0
        ranges = []
        for name, p in gm.named_parameters():
            is_mtp = ".mtp." in name or name.startswith("mtp.")
            mg = getattr(p, "main_grad", None)
            if is_mtp:
                mtp_total += 1
                if mg is not None:
                    mtp_has_main_grad += 1
                    nz = float(mg.detach().float().abs().sum().item())
                    if nz > 0:
                        mtp_nonzero.append((name, nz))
            if mg is not None:
                try:
                    base = mg.untyped_storage().data_ptr()
                    start = base + mg.storage_offset() * mg.element_size()
                    end = start + mg.numel() * mg.element_size()
                    ranges.append((name, is_mtp, start, end))
                except Exception:
                    pass

        aliases = []
        mtp_ranges = [r for r in ranges if r[1]]
        pol_ranges = [r for r in ranges if not r[1]]
        for mn, _, ms, me in mtp_ranges:
            for pn, _, ps, pe in pol_ranges:
                if ms < pe and ps < me:
                    aliases.append((mn, pn))
        return {
            "mtp_total_params": mtp_total,
            "mtp_params_with_main_grad": mtp_has_main_grad,
            "mtp_nonzero_after_zero_grad_buffer": mtp_nonzero[:10],
            "num_mtp_policy_grad_aliases": len(aliases),
            "alias_examples": aliases[:6],
        }


_CouplingPolicyWorker = ray.remote(num_gpus=1)(_CouplingWorker)


def _cfg(tp=4, packed=True):
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.placement.ref_num_gpus_per_node = 8
    cfg.trainer.algorithm.use_kl_loss = False  # no ref model needed for a grad probe
    cfg.trainer.remove_microbatch_padding = packed
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0
    cfg.trainer.policy.optimizer_config.lr = 1e-6
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.num_speculative_tokens = 1
    cfg.trainer.mtp.loss_type = "soft_ce"
    cfg.trainer.policy.megatron_config.mtp_loss_topk = 256
    validate_cfg(cfg)
    return cfg


@pytest.mark.megatron
def test_mtp_distributed_weight_coupling(ray_init_fixture):
    """At TP=4/DP=2: does adding the draft loss change the POLICY gradient?"""
    cfg = _cfg(tp=4, packed=True)

    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
        dp_size = policy.actor_infos[0].rank.dp_size
        batch = make_dummy_training_batch(batch_size=dp_size * 2, seq_len=320, num_actions=256)

        layout = ray.get(policy.async_run_ray_method("pass_through", "probe_buffer_layout"))
        print("\n===== buffer layout (rank0) =====")
        for k, v in layout[0].items():
            print(f"{k:40s}: {v}")

        # weight 0.0
        ray.get(policy.async_run_ray_method("pass_through", "set_mtp_weight", 0.0))
        ray.get(policy.async_run_ray_method("mesh", "forward_backward", data=batch))
        n0 = ray.get(policy.async_run_ray_method("pass_through", "snapshot_policy_grad", "w0"))

        # weight 0.5 (same fixed batch)
        ray.get(policy.async_run_ray_method("pass_through", "set_mtp_weight", 0.5))
        ray.get(policy.async_run_ray_method("mesh", "forward_backward", data=batch))
        nh = ray.get(policy.async_run_ray_method("pass_through", "snapshot_policy_grad", "whalf"))

        delta = ray.get(policy.async_run_ray_method("pass_through", "compare_snapshots", "w0", "whalf"))

        print("\n===== grad norms (per rank) =====")
        for r, (a, b) in enumerate(zip(n0, nh)):
            print(f"rank{r}: w0 {a}  |  whalf {b}")
        print("\n===== POLICY grad delta w0 -> whalf (per rank) =====")
        for r, d in enumerate(delta):
            print(f"rank{r}: {d}")
        print("====================================================")
    finally:
        _megatron_worker_mod.PolicyWorker = _orig

    worst = max(d["rel_l2_diff"] for d in delta)
    assert worst < 1e-4, (
        f"POLICY GRADIENT depends on the draft loss (rel_l2={worst:.3e}). The draft loss "
        f"is leaking onto the policy through the distributed grad path. Details: {delta}"
    )


@pytest.mark.megatron
@pytest.mark.parametrize("tp", [4, 1], ids=["tp4", "tp1"])
def test_mtp_draft_autograd_leak(ray_init_fixture, tp):
    """Localize the leak: does the draft loss reach the policy via AUTOGRAD (detach failing at TP>1)
    or via the grad machinery? Run at TP=4 (leaks) and TP=1 (clean) for contrast."""
    cfg = _cfg(tp=tp, packed=True)
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_draft_autograd_leak"))[0]
        print(f"\n===== draft autograd leak (tp={tp}) =====")
        import json

        print(json.dumps(res, indent=2, default=str))
        print("==========================================")
    finally:
        _megatron_worker_mod.PolicyWorker = _orig
    # Documenting behavior; not asserting so both tp values report.


@pytest.mark.megatron
def test_mtp_grad_breakdown(ray_init_fixture):
    """Pinpoint which MiMo MTP param carries the dominating gradient."""
    cfg = _cfg(tp=4, packed=True)
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_mtp_grad_breakdown"))[0]
        import json

        print("\n===== MTP grad breakdown =====")
        print(json.dumps(res, indent=2, default=str))
        print("==============================")
    finally:
        _megatron_worker_mod.PolicyWorker = _orig


@pytest.mark.megatron
@pytest.mark.parametrize("mtp_enabled", [True, False], ids=["mtp_on", "nosd"])
def test_mtp_fixed_batch_param_drift(ray_init_fixture, mtp_enabled):
    """NO-CLIP, fixed batch, K optim steps: does the POLICY parameter trajectory differ MTP-on vs
    nosd? If training is truly decoupled, the policy params evolve identically (the rollout is the
    only thing that could differ in the real loop, and it's removed here by using a fixed batch).
    Diff /tmp/pf_*.json: equal => training bitwise-clean; different => a real training channel."""
    import json

    cfg = _cfg(tp=4, packed=True)
    cfg.trainer.mtp.enabled = mtp_enabled
    if mtp_enabled:
        cfg.trainer.mtp.loss_weight = 0.5
    cfg.trainer.policy.optimizer_config.max_grad_norm = 1e9  # NO CLIP -> isolate from clip dilution
    validate_cfg(cfg)
    K = 20
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
        dp_size = policy.actor_infos[0].rank.dp_size
        batch = make_dummy_training_batch(batch_size=dp_size * 2, seq_len=320, num_actions=256)
        fps = [ray.get(policy.async_run_ray_method("pass_through", "policy_param_fingerprint"))[0]]
        ents = []
        for _ in range(K):
            res = ray.get(policy.async_run_ray_method("mesh", "forward_backward", data=batch))
            ray.get(policy.async_run_ray_method("pass_through", "optim_step"))
            ents.append(round(res[0].metrics.get("policy_entropy"), 6))
            fps.append(ray.get(policy.async_run_ray_method("pass_through", "policy_param_fingerprint"))[0])
        out = f"/tmp/pf_{'mtpon' if mtp_enabled else 'nosd'}.json"
        with open(out, "w") as f:
            json.dump({"entropy": ents, "param_fp": fps}, f)
        print(f"\n===== fixed-batch noclip param drift (mtp_enabled={mtp_enabled}) -> {out} =====")
        print("entropy:", ents)
        print("final policy param sumsq:", fps[-1]["sumsq"])
        print("==================================================")
    finally:
        _megatron_worker_mod.PolicyWorker = _orig


@pytest.mark.megatron
@pytest.mark.parametrize("mtp_enabled", [True, False], ids=["mtp_on", "nosd"])
def test_mtp_policy_logits_fingerprint(ray_init_fixture, mtp_enabled):
    """Compare full-precision policy logits on a fixed input, MTP-on vs nosd. Diff /tmp/lf_*.json."""
    import json

    cfg = _cfg(tp=4, packed=True)
    cfg.trainer.mtp.enabled = mtp_enabled
    validate_cfg(cfg)
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_policy_logits_fingerprint"))[0]
        out = f"/tmp/lf_{'mtpon' if mtp_enabled else 'nosd'}.json"
        with open(out, "w") as f:
            json.dump(res, f)
        print(f"\n===== policy logits fingerprint (mtp_enabled={mtp_enabled}) =====")
        print(json.dumps(res, indent=2))
        print("===================================================")
    finally:
        _megatron_worker_mod.PolicyWorker = _orig


@pytest.mark.megatron
@pytest.mark.parametrize("mtp_enabled", [True, False], ids=["mtp_on", "nosd"])
def test_mtp_export_fingerprint(ray_init_fixture, mtp_enabled):
    """Export-fingerprint the MAIN weights with MTP on vs off (nosd). Diff the two /tmp/fp_*.json
    to see if vLLM receives different main weights when MTP is enabled (the rollout channel)."""
    import json

    cfg = _cfg(tp=4, packed=True)
    cfg.trainer.mtp.enabled = mtp_enabled
    validate_cfg(cfg)
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_export_fingerprint"))[0]
        out = f"/tmp/fp_{'mtpon' if mtp_enabled else 'nosd'}.json"
        with open(out, "w") as f:
            json.dump(res, f)
        print(f"\n===== export fingerprint (mtp_enabled={mtp_enabled}) -> {out} =====")
        print(f"num_main_tensors={res['num_main_tensors']}")
        print("===================================================")
    finally:
        _megatron_worker_mod.PolicyWorker = _orig


@pytest.mark.megatron
@pytest.mark.parametrize("tp", [4, 1], ids=["tp4", "tp1"])
def test_mtp_export_coupling(ray_init_fixture, tp):
    """Does perturbing (drifting) the MTP head corrupt the exported MAIN-model weights synced to vLLM?"""
    cfg = _cfg(tp=tp, packed=True)
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_export_coupling"))[0]
        print(f"\n===== MTP export coupling (tp={tp}) =====")
        import json

        print(json.dumps(res, indent=2, default=str))
        print("==========================================")
    finally:
        _megatron_worker_mod.PolicyWorker = _orig
    assert res["num_main_tensors_changed_by_mtp_perturbation"] == 0, (
        f"WEIGHT-SYNC LEAK: drifting the MTP head changes {res['num_main_tensors_changed_by_mtp_perturbation']} "
        f"exported MAIN-model tensors -> vLLM rollout policy gets corrupted. {res['changed_examples']}"
    )


@pytest.mark.megatron
@pytest.mark.parametrize("tp", [4, 1], ids=["tp4", "tp1"])
def test_mtp_packed_draft_autograd_leak(ray_init_fixture, tp):
    """PACKED draft path (the real failing config) autograd leak check at TP=4 vs TP=1."""
    cfg = _cfg(tp=tp, packed=True)
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_packed_draft_autograd_leak"))[0]
        print(f"\n===== PACKED draft autograd leak (tp={tp}) =====")
        import json

        print(json.dumps(res, indent=2, default=str))
        print("=================================================")
    finally:
        _megatron_worker_mod.PolicyWorker = _orig


@pytest.mark.megatron
@pytest.mark.parametrize("mtp_enabled", [True, False], ids=["mtp_on", "mtp_off"])
def test_mtp_presence_entropy_trajectory(ray_init_fixture, mtp_enabled):
    """Generation-free repro: run K real training steps (forward_backward + optim_step) on a
    FIXED batch at TP=4/DP=2 and log policy_entropy + grad_norm each step. Compare the mtp_on vs
    mtp_off trajectories. If mtp_on diverges on identical data + init + seed, enabling MTP is
    corrupting the policy update (the collapse channel), with NO rollout loop involved."""
    cfg = _cfg(tp=4, packed=True)
    cfg.trainer.mtp.enabled = mtp_enabled
    if mtp_enabled:
        cfg.trainer.mtp.loss_weight = 0.5  # the failing run's weight
    validate_cfg(cfg)

    K = 30
    policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
    dp_size = policy.actor_infos[0].rank.dp_size
    batch = make_dummy_training_batch(batch_size=dp_size * 2, seq_len=320, num_actions=256)

    print(f"\n===== entropy trajectory (mtp_enabled={mtp_enabled}) =====")
    for step in range(K):
        res = ray.get(policy.async_run_ray_method("mesh", "forward_backward", data=batch))
        gn = ray.get(policy.async_run_ray_method("pass_through", "optim_step"))
        m0 = res[0].metrics
        print(
            f"step {step:3d} | entropy {m0.get('policy_entropy'):.5f} | "
            f"policy_loss {m0.get('policy_loss'):.5f} | mtp_loss {m0.get('mtp_loss')} | "
            f"grad_norm {gn[0]:.4f}",
            flush=True,
        )
    print("=========================================================")


@pytest.mark.megatron
def test_cfull_buffer_isolation(ray_init_fixture):
    """C-full correctness: the policy DDP grad buffer must EXCLUDE every .mtp param (so the policy
    reduction is byte-identical to a no-MTP model), and the MTP head must live in its own buffer."""
    cfg = _cfg(tp=4, packed=True)
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.loss_weight = 0.5
    validate_cfg(cfg)
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_cfull_isolation"))[0]
    finally:
        _megatron_worker_mod.PolicyWorker = _orig
    import json

    print("\n===== C-full buffer isolation =====")
    print(json.dumps(res, indent=2, default=str))
    print("===================================")
    assert res["mtp_separate_built"], "C-full did not build the separate MTP optimizer"
    assert res["mtp_total"] > 0, "no MTP params found"
    assert res["mtp_in_POLICY_buffer"] == 0, (
        f"C-full LEAK: {res['mtp_in_POLICY_buffer']}/{res['mtp_total']} MTP params are still in the "
        f"POLICY grad buffer -> policy reduction NOT byte-identical to no-MTP. {res}"
    )
    assert res["mtp_in_MTP_buffer"] == res["mtp_total"], (
        f"C-full: only {res['mtp_in_MTP_buffer']}/{res['mtp_total']} MTP params landed in the separate "
        f"MTP buffer. {res}"
    )


@pytest.mark.megatron
@pytest.mark.parametrize("mtp_enabled", [True, False], ids=["cfull_mtp_on", "nosd"])
def test_cfull_policy_gradnorm(ray_init_fixture, mtp_enabled):
    """Decisive: does C-full's reported POLICY grad_norm match the no-MTP baseline, or is it inflated
    (head leaking into the policy clip)? Same fixed dummy batch both arms. If cfull_mtp_on's
    optim_step grad_norm >> nosd's, the policy clip is still diluted by the head (the bug that would
    re-create the collapse). The manual breakdown shows the head's raw main_grad (huge) vs policy."""
    cfg = _cfg(tp=4, packed=True)
    cfg.trainer.mtp.enabled = mtp_enabled
    if mtp_enabled:
        cfg.trainer.mtp.loss_weight = 0.5
    cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0
    validate_cfg(cfg)
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _CouplingPolicyWorker
    try:
        policy = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, cfg=cfg
        )
        dp = policy.actor_infos[0].rank.dp_size
        batch = make_dummy_training_batch(batch_size=dp * 2, seq_len=320, num_actions=256)
        ray.get(policy.async_run_ray_method("mesh", "forward_backward", data=batch))
        snap = ray.get(policy.async_run_ray_method("pass_through", "snapshot_policy_grad", "s"))[0]
        gn = ray.get(policy.async_run_ray_method("pass_through", "optim_step"))[0]
    finally:
        _megatron_worker_mod.PolicyWorker = _orig
    tag = "cfull_mtp_on" if mtp_enabled else "nosd"
    print(f"\n===== policy grad_norm [{tag}] =====")
    print(f"manual main_grad (rank0): policy={snap['policy_grad_norm']:.5f}  mtp={snap['mtp_grad_norm']:.5f}")
    print(f"optim_step reported POLICY grad_norm = {gn}")
    print("=====")
