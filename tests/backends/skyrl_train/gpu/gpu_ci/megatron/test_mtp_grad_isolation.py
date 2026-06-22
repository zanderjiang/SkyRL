"""Decisive isolation probe: does the decoupled MTP/draft loss leak gradient onto
the policy trunk/embedding/output head?

The design intends the draft loss to train ONLY ``.mtp.*`` params (trunk hidden +
embedding + output weight all detached, teacher detached). This probe backprops
ONLY the draft loss on the REAL failing model (MiMo-7B-RL, untied embeddings) with
the REAL loss config (soft-CE top-k=256, mtp_detach_shared_output=True) and reports
the grad norm of every parameter, partitioned into ``.mtp.*`` vs everything else.

Expected if isolation holds:
  - some ``.mtp.*`` params have grad > 0   (the head actually trains)
  - EVERY non-mtp param has grad None or ~0 (no leak onto the policy)

Any non-mtp param with nonzero grad is a leak that would let the draft loss reshape
the policy itself (a candidate cause of the spec-decode entropy collapse).

Run::
    uv run --isolated --extra megatron --extra dev pytest -s -vvv \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_mtp_grad_isolation.py
"""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as _megatron_worker_mod,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

MODEL_NAME = "XiaomiMiMo/MiMo-7B-RL"


class _ProbeMegatronPolicyWorker(MegatronPolicyWorkerBase):
    def probe_grad_isolation(self, token_ids=None, seq_len: int = 96, topk: int = 256) -> dict:
        import torch
        from megatron.core import parallel_state as mpu
        from megatron.core.utils import unwrap_model

        from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
        from skyrl.backends.skyrl_train.mtp.hidden_capture import (
            MTPHiddenCapture,
            _resolve_mtp_host,
            _unwrap_model,
        )
        from skyrl.backends.skyrl_train.mtp.soft_ce import draft_soft_ce_topk, shift_mask_for_mtp

        gm = unwrap_model(self.actor_module[0])
        host = _resolve_mtp_host(_unwrap_model(self.actor_module[0]))
        if getattr(host, "mtp", None) is None:
            return {"error": "no mtp head built", "host": type(host).__name__}

        device = torch.cuda.current_device()
        if token_ids is not None:
            ids = torch.tensor(token_ids, device=device, dtype=torch.long)
            seq_len = ids.shape[0]
        else:
            ids = (torch.arange(seq_len, device=device) * 7 + 3) % 100000
        sequences = ids.unsqueeze(0)  # [1, S]
        attention_mask = torch.ones_like(sequences)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Faithful to training: detach_trunk + detach_shared_embedding == mtp_detach_shared_output=True.
        capture = MTPHiddenCapture(self.actor_module[0], detach_trunk=True, detach_shared_embedding=True)

        # Forward WITH grad (so a graph exists), capture + replay the MTP head.
        with capture.capture():
            outputs = self.actor_module[0](sequences, position_ids, attention_mask)
            student_hidden = capture.compute_student_hidden_states()  # list per depth

        def to_bsv(x):
            if x.dim() != 3:
                return x
            if x.shape[0] == 1:
                return x
            if x.shape[1] == 1:
                return x.transpose(0, 1).contiguous()
            return x

        # Detached output weight == mtp_detach_shared_output=True (untied MiMo -> output_layer.weight).
        student_logits = project_mtp_hidden_to_logits(student_hidden, host, detach_output_weight=True)[0]
        student_logits = to_bsv(student_logits)  # [B, S, V/tp]
        teacher_logits = to_bsv(outputs)  # main policy logits; draft_soft_ce_topk detaches internally

        tp_grp = mpu.get_tensor_model_parallel_group()
        mask = torch.ones(student_logits.shape[0], student_logits.shape[1], device=device)
        layer_mask = shift_mask_for_mtp(mask, 0)  # depth-0 validity (unpacked global roll)

        draft_loss = draft_soft_ce_topk(
            student_logits,
            teacher_logits,
            layer_mask,
            k=topk,
            vocab_parallel_group=tp_grp,
            roll_shift=1,
        )

        # Backprop ONLY the draft loss. Megatron DDP accumulates into param.main_grad (a pre-allocated
        # grad buffer), NOT param.grad, so zero the buffer and read main_grad below.
        try:
            self.actor_module[0].zero_grad_buffer()
        except Exception:
            pass
        self.actor_module[0].zero_grad(set_to_none=True)
        connected = bool(getattr(student_logits, "requires_grad", False)) and (draft_loss.grad_fn is not None)
        draft_loss.backward()

        def _grad_of(p):
            # Prefer Megatron's main_grad buffer; fall back to .grad.
            mg = getattr(p, "main_grad", None)
            if mg is not None:
                return mg
            return p.grad

        # Partition every param by whether it belongs to the MTP head (".mtp." in the qualified name).
        mtp_with_grad, mtp_total = 0, 0
        nonmtp_with_grad, nonmtp_total = 0, 0
        leak_top = []  # (name, grad_norm) for non-mtp params with nonzero grad
        mtp_grad_norm_sq = 0.0
        # Specifically watch the shared-by-design params (untied: separate, must stay at 0 grad).
        watched = {}
        for name, p in gm.named_parameters():
            is_mtp = ".mtp." in name or name.startswith("mtp.")
            g = _grad_of(p)
            gnorm = float(g.detach().float().norm().item()) if g is not None else 0.0
            if is_mtp:
                mtp_total += 1
                if gnorm > 0:
                    mtp_with_grad += 1
                    mtp_grad_norm_sq += gnorm * gnorm
            else:
                nonmtp_total += 1
                if gnorm > 1e-12:
                    nonmtp_with_grad += 1
                    leak_top.append((name, gnorm))
            low = name.lower()
            if any(t in low for t in ("output_layer", "embedding", "word_embeddings", "lm_head")) and "mtp" not in low:
                watched[name] = gnorm

        leak_top.sort(key=lambda kv: kv[1], reverse=True)
        return {
            "host": type(host).__name__,
            "draft_loss": float(draft_loss.detach().item()),
            "graph_connected(student.requires_grad & loss.grad_fn)": connected,
            "seq_len": int(seq_len),
            "vocab_shard": tuple(student_logits.shape),
            "mtp_params_with_grad": f"{mtp_with_grad}/{mtp_total}",
            "mtp_grad_global_norm": float(mtp_grad_norm_sq**0.5),
            "nonmtp_params_with_grad": f"{nonmtp_with_grad}/{nonmtp_total}",
            "LEAK_top10": leak_top[:10],
            "watched_shared_params_gradnorm": watched,
        }


    def probe_packed_grad_isolation(self, token_ids=None, topk: int = 256) -> dict:
        """Same isolation check but through the THD PACKED path (remove_microbatch_padding=True),
        the configuration that collapses. Packs TWO sub-sequences into one row so multi-segment
        packing is exercised, runs the real preprocess_packed_seqs -> packed forward -> capture/replay
        -> packed draft loss, backprops ONLY the draft loss, and checks main_grad on every param.
        Also verifies the replay does not mutate the main (policy) logits in place."""
        import torch
        from megatron.core import parallel_state as mpu
        from megatron.core.utils import unwrap_model

        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import preprocess_packed_seqs
        from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
        from skyrl.backends.skyrl_train.mtp.hidden_capture import (
            MTPHiddenCapture,
            _resolve_mtp_host,
            _unwrap_model,
        )
        from skyrl.backends.skyrl_train.mtp.soft_ce import draft_soft_ce_topk, shift_mask_for_mtp
        from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import _build_packed_valid_mask

        gm = unwrap_model(self.actor_module[0])
        host = _resolve_mtp_host(_unwrap_model(self.actor_module[0]))
        if getattr(host, "mtp", None) is None:
            return {"error": "no mtp head built"}

        device = torch.cuda.current_device()
        # Two sub-sequences of different real lengths, LEFT-padded into a [2, S] batch (like real RL
        # batches), so preprocess_packed_seqs packs them into one [1, T] row with >1 segment.
        ids = token_ids if token_ids is not None else list(range(3, 99))
        a = ids[:60]
        b = ids[:40]
        S = 60
        pad_id = 0

        def leftpad(x):
            return [pad_id] * (S - len(x)) + x

        sequences = torch.tensor([leftpad(a), leftpad(b)], device=device, dtype=torch.long)  # [2, S]
        attention_mask = torch.tensor(
            [[0] * (S - len(a)) + [1] * len(a), [0] * (S - len(b)) + [1] * len(b)], device=device, dtype=torch.bool
        )

        new_sequences, packed_seq_params = preprocess_packed_seqs(
            sequences, attention_mask, pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True)
        )

        capture = MTPHiddenCapture(self.actor_module[0], detach_trunk=True, detach_shared_embedding=True)
        with capture.capture():
            outputs = self.actor_module[0](new_sequences, None, None, packed_seq_params=packed_seq_params)
            logits_before = outputs.detach().clone()
            student_hidden = capture.compute_student_hidden_states()

        # Integrity: did the decoupled replay mutate the policy logits in place?
        logits_mutated_by_replay = not torch.equal(outputs.detach(), logits_before)

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

        try:
            self.actor_module[0].zero_grad_buffer()
        except Exception:
            pass
        self.actor_module[0].zero_grad(set_to_none=True)
        connected = bool(getattr(student_logits, "requires_grad", False)) and (draft_loss.grad_fn is not None)
        draft_loss.backward()

        def _grad_of(p):
            mg = getattr(p, "main_grad", None)
            return mg if mg is not None else p.grad

        mtp_with_grad = mtp_total = nonmtp_with_grad = nonmtp_total = 0
        leak_top = []
        watched = {}
        for name, p in gm.named_parameters():
            is_mtp = ".mtp." in name or name.startswith("mtp.")
            g = _grad_of(p)
            gnorm = float(g.detach().float().norm().item()) if g is not None else 0.0
            if is_mtp:
                mtp_total += 1
                mtp_with_grad += int(gnorm > 0)
            else:
                nonmtp_total += 1
                if gnorm > 1e-12:
                    nonmtp_with_grad += 1
                    leak_top.append((name, gnorm))
            low = name.lower()
            if any(t in low for t in ("output_layer", "embedding", "word_embeddings", "lm_head")) and "mtp" not in low:
                watched[name] = gnorm
        leak_top.sort(key=lambda kv: kv[1], reverse=True)
        return {
            "path": "PACKED (remove_microbatch_padding=True)",
            "draft_loss": float(draft_loss.detach().item()),
            "graph_connected": connected,
            "num_packed_segments": int(cu.numel() - 1),
            "packed_T": int(new_sequences.shape[1]) if new_sequences.dim() == 2 else int(new_sequences.shape[0]),
            "logits_mutated_by_replay": logits_mutated_by_replay,
            "mtp_params_with_grad": f"{mtp_with_grad}/{mtp_total}",
            "nonmtp_params_with_grad": f"{nonmtp_with_grad}/{nonmtp_total}",
            "LEAK_top10": leak_top[:10],
            "watched_shared_params_gradnorm": watched,
        }


    def probe_component_grads(self, token_ids=None, seq_len: int = 96) -> dict:
        """Measure where gradient actually flows in a realistic backward. On the MTP-enabled worker
        with a FIXED input, backprop three losses separately and report grad norms grouped by
        trunk / mtp-head / embedding / output_layer:
          (a) policy CE on the MAIN logits alone  -> trunk should get grad, mtp should be ~0
          (b) draft soft-CE alone                 -> mtp should get grad, trunk ~0
          (c) policy CE + 0.5*draft               -> trunk grad must equal (a) by linearity
        If (c)'s trunk grad != (a)'s, or (a) gives the mtp head grad, there is real coupling."""
        import torch
        from megatron.core import parallel_state as mpu
        from megatron.core.utils import unwrap_model

        from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
        from skyrl.backends.skyrl_train.mtp.hidden_capture import MTPHiddenCapture, _resolve_mtp_host, _unwrap_model
        from skyrl.backends.skyrl_train.mtp.soft_ce import draft_soft_ce_topk, shift_mask_for_mtp

        gm = unwrap_model(self.actor_module[0])
        host = _resolve_mtp_host(_unwrap_model(self.actor_module[0]))
        if getattr(host, "mtp", None) is None:
            return {"error": "no mtp head built"}
        device = torch.cuda.current_device()
        if token_ids is not None:
            ids = torch.tensor(token_ids, device=device, dtype=torch.long)
            seq_len = ids.shape[0]
        else:
            ids = (torch.arange(seq_len, device=device) * 7 + 3) % 100000
        sequences = ids.unsqueeze(0)
        attention_mask = torch.ones_like(sequences)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        tp_grp = mpu.get_tensor_model_parallel_group()

        def to_bsv(x):
            if x.dim() == 3 and x.shape[0] != 1 and x.shape[1] == 1:
                return x.transpose(0, 1).contiguous()
            return x

        def group_grads():
            g = {"mtp": 0.0, "trunk": 0.0, "embedding": 0.0, "output_layer": 0.0}
            for name, p in gm.named_parameters():
                mg = getattr(p, "main_grad", None)
                val = mg if mg is not None else p.grad
                n = float(val.detach().float().norm().item()) ** 2 if val is not None else 0.0
                low = name.lower()
                if ".mtp." in name or name.startswith("mtp."):
                    g["mtp"] += n
                elif ("embedding" in low or "word_embeddings" in low) and "mtp" not in low:
                    g["embedding"] += n
                elif "output_layer" in low and "mtp" not in low:
                    g["output_layer"] += n
                else:
                    g["trunk"] += n
            return {k: v**0.5 for k, v in g.items()}

        def zero():
            try:
                self.actor_module[0].zero_grad_buffer()
            except Exception:
                pass
            self.actor_module[0].zero_grad(set_to_none=True)

        def policy_ce():
            # CE of main logits vs next token (proxy for the policy loss that trains the trunk)
            with MTPHiddenCapture(self.actor_module[0], True, True).capture():
                out = self.actor_module[0](sequences, position_ids, attention_mask)
            ml = to_bsv(out).float()
            lp = torch.log_softmax(ml[:, :-1], dim=-1)
            tgt = sequences[:, 1:]
            return -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()

        def draft():
            cap = MTPHiddenCapture(self.actor_module[0], True, True)
            with cap.capture():
                out = self.actor_module[0](sequences, position_ids, attention_mask)
                sh = cap.compute_student_hidden_states()
            st = to_bsv(project_mtp_hidden_to_logits(sh, host, detach_output_weight=True)[0])
            te = to_bsv(out)
            m = shift_mask_for_mtp(torch.ones(st.shape[0], st.shape[1], device=device), 0)
            return draft_soft_ce_topk(st, te, m, k=256, vocab_parallel_group=tp_grp, roll_shift=1)

        # DEFINITIVE check: does the policy loss (CE on main logits) actually depend on MTP params?
        # Use autograd.grad directly (bypasses main_grad/zeroing entirely).
        out_nc = self.actor_module[0](sequences, position_ids, attention_mask)
        ml_nc = to_bsv(out_nc).float()
        lp_nc = torch.log_softmax(ml_nc[:, :-1], dim=-1)
        pce_nc = -lp_nc.gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1).mean()
        mtp_named = [(n, p) for n, p in gm.named_parameters() if (".mtp." in n or n.startswith("mtp.")) and p.requires_grad]
        mtp_params = [p for _, p in mtp_named]
        grads = torch.autograd.grad(pce_nc, mtp_params, allow_unused=True, retain_graph=False)
        nonzero = [(mtp_named[i][0], float(g.detach().float().norm().item())) for i, g in enumerate(grads) if g is not None]
        autograd_check = {
            "num_mtp_params": len(mtp_params),
            "num_mtp_params_in_policy_graph": len(nonzero),
            "policy_loss_depends_on_ANY_mtp_param": len(nonzero) > 0,
            "nonzero_examples": nonzero[:8],
        }

        def policy_ce_nocapture():
            out = self.actor_module[0](sequences, position_ids, attention_mask)
            ml = to_bsv(out).float()
            lp = torch.log_softmax(ml[:, :-1], dim=-1)
            return -lp.gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1).mean()

        # (a0) policy CE WITHOUT the capture context (plain model forward) — does main forward still
        # depend on the MTP head?
        zero(); policy_ce_nocapture().backward(); ga0 = group_grads()
        # (a) policy CE alone (inside capture, as training runs it)
        zero(); policy_ce().backward(); ga = group_grads()
        # (b) draft alone
        zero(); draft().backward(); gb = group_grads()
        # (c) combined
        zero()
        cap = MTPHiddenCapture(self.actor_module[0], True, True)
        with cap.capture():
            out = self.actor_module[0](sequences, position_ids, attention_mask)
            sh = cap.compute_student_hidden_states()
        ml = to_bsv(out).float()
        lp = torch.log_softmax(ml[:, :-1], dim=-1)
        pce = -lp.gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1).mean()
        st = to_bsv(project_mtp_hidden_to_logits(sh, host, detach_output_weight=True)[0])
        m = shift_mask_for_mtp(torch.ones(st.shape[0], st.shape[1], device=device), 0)
        dl = draft_soft_ce_topk(st, to_bsv(out), m, k=256, vocab_parallel_group=tp_grp, roll_shift=1)
        (pce + 0.5 * dl).backward()
        gc = group_grads()

        return {
            "AUTOGRAD_CHECK_policy_depends_on_mtp": autograd_check,
            "a0_policy_ce_NO_capture": ga0,
            "a_policy_ce_alone": ga,
            "b_draft_alone": gb,
            "c_policy_plus_0.5draft": gc,
            "trunk_grad_changed_by_draft": abs(gc["trunk"] - ga["trunk"]) > 1e-6 * max(1.0, ga["trunk"]),
            "emb_grad_changed_by_draft": abs(gc["embedding"] - ga["embedding"]) > 1e-6 * max(1.0, ga["embedding"]),
        }


_ProbePolicyWorker = ray.remote(num_gpus=1)(_ProbeMegatronPolicyWorker)


def _cfg():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    # Match the failing run's MTP loss config.
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.num_speculative_tokens = 1
    cfg.trainer.mtp.loss_type = "soft_ce"
    cfg.trainer.policy.megatron_config.mtp_loss_topk = 256
    validate_cfg(cfg)
    return cfg


@pytest.mark.megatron
def test_mtp_grad_isolation(ray_init_fixture):
    cfg = _cfg()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The derivative of x squared is two x. In 1969, "
        "Apollo 11 landed the first humans on the Moon. Water boils at one hundred degrees Celsius "
        "at sea level. The capital of France is Paris, a city on the river Seine."
    )
    token_ids = tok(passage, add_special_tokens=False)["input_ids"][:128]

    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _ProbePolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=1, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_grad_isolation", token_ids))[0]
    finally:
        _megatron_worker_mod.PolicyWorker = _orig

    print("\n===== MTP grad isolation =====")
    for k, v in res.items():
        print(f"{k:32s}: {v}")
    print("==============================")

    assert "error" not in res, res
    # The head must actually train...
    assert res["mtp_params_with_grad"].split("/")[0] != "0", f"draft loss trained no MTP params: {res}"
    # ...and NOTHING else may receive gradient from the draft loss.
    assert res["nonmtp_params_with_grad"].startswith("0/"), (
        f"GRADIENT LEAK: the draft loss reaches non-MTP params -> it can reshape the policy. "
        f"Top offenders: {res['LEAK_top10']}"
    )


@pytest.mark.megatron
def test_mtp_component_grads(ray_init_fixture):
    cfg = _cfg()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    token_ids = tok("The capital of France is Paris. Water boils at 100 degrees. Two plus two is four.",
                    add_special_tokens=False)["input_ids"][:96]
    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _ProbePolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=1, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_component_grads", token_ids))[0]
    finally:
        _megatron_worker_mod.PolicyWorker = _orig
    print("\n===== MTP component grads =====")
    import json
    print(json.dumps(res, indent=2, default=str))
    print("===============================")
    assert "error" not in res, res


@pytest.mark.megatron
def test_mtp_grad_isolation_packed(ray_init_fixture):
    cfg = _cfg()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The derivative of x squared is two x. In 1969, "
        "Apollo 11 landed the first humans on the Moon."
    )
    token_ids = tok(passage, add_special_tokens=False)["input_ids"][:96]

    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _ProbePolicyWorker
    try:
        policy = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=1, cfg=cfg)
        res = ray.get(policy.async_run_ray_method("pass_through", "probe_packed_grad_isolation", token_ids))[0]
    finally:
        _megatron_worker_mod.PolicyWorker = _orig

    print("\n===== MTP grad isolation (PACKED) =====")
    for k, v in res.items():
        print(f"{k:32s}: {v}")
    print("=======================================")

    assert "error" not in res, res
    assert res["mtp_params_with_grad"].split("/")[0] != "0", f"draft loss trained no MTP params: {res}"
    assert not res["logits_mutated_by_replay"], "decoupled replay mutated the policy logits in place (packed)!"
    assert res["nonmtp_params_with_grad"].startswith("0/"), (
        f"PACKED GRADIENT LEAK: draft loss reaches non-MTP params. Top offenders: {res['LEAK_top10']}"
    )
