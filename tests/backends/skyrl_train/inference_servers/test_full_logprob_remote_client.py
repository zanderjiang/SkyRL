import numpy as np
import pytest

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    PackedField,
    pack_full_logprobs,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteGenerateClient,
)


class _Entry:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_logprobs", [None, 0, 1])
async def test_requests_and_decodes_full_logprobs(monkeypatch, requested_logprobs):
    client = RemoteGenerateClient(proxy_url="http://unused")
    captured = {}
    values = np.array([[-3.0, -0.25, -2.0]], dtype=np.float32)
    rows = [{token_id: _Entry(float(value)) for token_id, value in enumerate(values[0])}]

    async def fake_post(url, json, headers, *, packed_side_channels=False):
        captured.update(url=url, json=json, packed_side_channels=packed_side_channels)
        return {
            "choices": [
                {
                    "token_ids": [1],
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"logprob": -0.25}]},
                    PackedField.FULL_LOGPROBS.value: pack_full_logprobs([1], rows, vocab_size=3),
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)
    result = await client.generate(
        prompt_token_ids=[1, 2],
        sampling_params={"temperature": 1.0, "logprobs": requested_logprobs},
        session_id=None,
        model="default",
        return_full_logprobs=True,
    )

    assert captured["url"].endswith("/skyrl/v1/generate")
    assert captured["json"]["return_full_logprobs"] is True
    assert captured["json"]["sampling_params"]["logprobs"] == -1
    assert captured["json"]["sampling_params"]["detokenize"] is False
    assert captured["packed_side_channels"] is True
    assert result.full_logprobs.tobytes() == values.tobytes()


@pytest.mark.asyncio
async def test_zero_logprobs_keeps_sampled_action_score(monkeypatch):
    client = RemoteGenerateClient(proxy_url="http://unused")
    captured = {}

    async def fake_post(url, json, headers, *, packed_side_channels=False):
        captured.update(url=url, json=json)
        return {
            "choices": [
                {
                    "token_ids": [7],
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"logprob": -0.25}]},
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)
    params = {"temperature": 1.0, "logprobs": 0}
    result = await client.generate(
        prompt_token_ids=[1, 2],
        sampling_params=params,
        session_id=None,
        model="default",
    )
    assert captured["url"].endswith("/inference/v1/generate")
    assert captured["json"]["sampling_params"]["logprobs"] == 0
    assert params == {"temperature": 1.0, "logprobs": 0}
    assert result.response_logprobs == [-0.25]
