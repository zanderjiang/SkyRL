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
async def test_requests_and_decodes_full_logprobs(monkeypatch):
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
        sampling_params={"temperature": 1.0},
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
