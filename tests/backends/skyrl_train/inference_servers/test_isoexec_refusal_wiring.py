from unittest.mock import AsyncMock, Mock

import pytest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("isoexec", [False, True])
async def test_request_identity_preserves_native_generate_route(isoexec):
    client = RemoteInferenceClient(
        proxy_url="http://router",
        server_urls=["http://engine"],
        data_parallel_size=1,
        verify_isoexec_weights=isoexec,
    )
    client._post = AsyncMock(
        return_value={"choices": [{"token_ids": [5], "finish_reason": "length"}]}
    )
    result = await client._generate_single([1, 2], {}, "prompt_uid_3", "policy")
    call = client._post.await_args
    assert call.args[0] == "http://router/inference/v1/generate"
    assert result["response_ids"] == [5]
    if isoexec:
        assert call.kwargs["json"]["request_id"].split("--")[0] == "prompt_uid_3"
        assert (
            call.kwargs["json"]["sampling_params"]["extra_args"]["isoexec_request_id"]
            == call.kwargs["json"]["request_id"]
        )
        assert (
            call.kwargs["headers"]["X-Request-Id"] == call.kwargs["json"]["request_id"]
        )
    else:
        assert "request_id" not in call.kwargs["json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["clean", "corrupted-during-generation"])
async def test_step_capture_collects_every_engine_rank(tmp_path, monkeypatch, verdict):
    import json
    from pathlib import Path

    from isoexec.integrations.skyrl.audit import record_request

    monkeypatch.setenv("ISOEXEC_REFUSAL_ROOT", str(tmp_path))
    client = RemoteInferenceClient(
        proxy_url="http://router",
        server_urls=["http://engine0", "http://engine1"],
        data_parallel_size=1,
        verify_isoexec_weights=True,
    )
    ranks = [
        {
            "artifact_dir": f"/capture/rank{i}",
            "verdict": "clean",
            "trace": [
                {
                    "request_aliases": {
                        "internal-kept": "kept--1",
                        "internal-discarded": "discarded--2",
                    }
                }
            ],
        }
        for i in range(4)
    ]
    ranks[0]["verdict"] = verdict
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    @asynccontextmanager
    async def respond(method, url, **kwargs):
        results = ranks[:2] if url.startswith("http://engine0/") else ranks[2:]
        yield SimpleNamespace(
            content_length=1,
            status=200,
            json=AsyncMock(return_value={"results": results}),
            raise_for_status=lambda: None,
        )

    session = SimpleNamespace(request=Mock(side_effect=respond))
    client._get_session = AsyncMock(return_value=session)
    for request_id in ("kept--1", "discarded--2"):
        await client.isoexec_refusal_begin_step("wire-test", 7, {})
        record_request(
            client,
            {"token_ids": [1, 2], "request_id": request_id},
            {"id": request_id, "choices": [{"token_ids": [3, 4]}]},
            request_id.split("--")[0],
        )
    session.request.reset_mock()
    assert await client.isoexec_refusal_end_step() == ranks
    records = [
        json.loads(line)
        for line in Path(ranks[0]["request_outputs"]).read_text().splitlines()
    ]
    assert [record["request_id"] for record in records] == [
        "internal-kept",
        "internal-discarded",
    ]
    assert (Path(ranks[0]["request_outputs"]).parent / "mismatch").exists() == (
        verdict != "clean"
    )
    assert session.request.call_count == 2
    assert {call.args[:2] for call in session.request.call_args_list} == {
        ("POST", "http://engine0/collective_rpc"),
        ("POST", "http://engine1/collective_rpc"),
    }
    assert all(
        call.kwargs["json"] == {"method": "isoexec_refusal_end_step"}
        for call in session.request.call_args_list
    )


def test_trainer_gate_retains_batch_alignment_and_exports_before_refusal(
    tmp_path, monkeypatch
):
    import json

    import torch
    from isoexec.checks import Refusal

    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.trainer import RayPPOTrainer

    monkeypatch.setenv("ISOEXEC_REFUSAL_ROOT", str(tmp_path))
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = SkyRLTrainConfig()
    trainer.cfg.trainer.enable_isoexec = True
    trainer.cfg.trainer.critic.model.path = None
    trainer.cfg.trainer.run_name = "wire-test"
    trainer.global_step = 7
    trainer._isoexec_engine_artifacts = []
    trainer.ref_model = None
    trainer.dispatch = Mock()
    trainer_artifacts = tmp_path / "trainer-artifacts"
    trainer_artifacts.mkdir()
    trainer.dispatch.isoexec_refusal_receipts.return_value = [
        {
            "weight_bytes": 8,
            "trainer_weights": {
                "artifact_dir": str(trainer_artifacts),
                "verdict": "clean",
            },
        }
    ]
    trainer._execute_forward_pass = Mock(
        return_value=torch.tensor([[0.0, -1.0, -2.0], [-1.0, -2.0, -3.0]])
    )
    exports = []

    def save_weights(path):
        exports.append(path)
        torch.save({"weight": torch.tensor([1.0, 2.0])}, path / "model.pt")

    trainer._isoexec_save_refusal_weights = save_weights
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            "response_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "loss_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "rollout_logprobs": torch.tensor([[0.0, -1.0, -2.0], [-1.0, -1.75, -2.5]]),
        }
    )
    batch.metadata = {
        "response_length": 3,
        "uids": ["a", "b"],
        "request_sessions": ["a_0", "b_0"],
        "pad_size": 0,
    }
    with pytest.raises(Refusal, match="exact-zero pre-update"):
        trainer.fwd_logprobs_values_reward(batch)

    bundle = next((tmp_path / "wire-test").iterdir())
    saved = torch.load(bundle / "batch.pt", weights_only=False)
    assert saved["rollout_logprobs"].shape == (2, 3)
    assert torch.equal(saved["sequences"], batch["sequences"])
    assert saved["request_sessions"] == ["a_0", "b_0"]
    assert saved["prompt_lengths"].tolist() == [2, 2]
    divergence = json.loads((bundle / "divergence.json").read_text())
    assert divergence[0]["batch_idx"] == 1
    assert divergence[0]["onset_position"] == 3
    assert divergence[0]["run_length"] == 2
    assert exports == [bundle / "weights"]


@pytest.mark.asyncio
@pytest.mark.parametrize("active_capture", [False, True])
async def test_batched_generator_carries_unique_sessions_only_during_capture(
    monkeypatch, active_capture
):
    from types import SimpleNamespace

    from skyrl.train.generators import skyrl_gym_generator as module

    generator = module.SkyRLGymGenerator.__new__(module.SkyRLGymGenerator)
    generator.batched = True
    generator.max_turns = 1
    generator.policy_model_name = "policy"
    generator.skyrl_gym_cfg = SimpleNamespace()
    generator.generator_cfg = SimpleNamespace(
        step_wise_trajectories=False,
        apply_overlong_filtering=False,
        max_input_length=32,
        sampling_params=SimpleNamespace(max_generate_length=8),
    )
    generator.tokenizer = SimpleNamespace(
        apply_chat_template=Mock(return_value=[[1, 2], [1, 2]])
    )
    generator._compute_cache_salt = Mock(return_value=None)
    generator.inference_engine_client = SimpleNamespace(
        _isoexec_requests=active_capture,
        generate=AsyncMock(
            return_value={
                "responses": ["a", "b"],
                "response_ids": [[3], [4]],
                "stop_reasons": ["length", "length"],
            }
        ),
    )
    env = SimpleNamespace(
        init=lambda prompt: (prompt, None),
        step=lambda output: {"reward": 1},
        get_metrics=dict,
        close=lambda: None,
    )
    monkeypatch.setattr(module.skyrl_gym, "make", lambda *a, **k: env)
    monkeypatch.setattr(module, "get_rollout_metrics", lambda *a: {})

    async def invoke(fn, *args):
        return fn(*args)

    generator._run_in_executor_if_available = invoke
    tids = [
        SimpleNamespace(instance_id="same-prompt", repetition_id=i) for i in range(2)
    ]
    await generator.generate(
        {
            "prompts": [[], []],
            "env_classes": ["test", "test"],
            "env_extras": [{}, {}],
            "trajectory_ids": tids,
            "sampling_params": {},
        }
    )
    engine_input = generator.inference_engine_client.generate.await_args.args[0]
    assert engine_input["session_ids"] == (
        ["same-prompt_0", "same-prompt_1"] if active_capture else None
    )
