"""The native sampler receives the configured minimum length without overriding it."""

from dataclasses import asdict

import pytest
from omegaconf import OmegaConf
from vllm import SamplingParams as VllmSamplingParams

from skyrl.backends.skyrl_train.inference_servers.engine_utils import get_vllm_sampling_params
from skyrl.train.config import SamplingParams


@pytest.mark.parametrize("minimum", [None, 0, 3])
@pytest.mark.parametrize("legacy_config", [False, True])
def test_minimum_length_reaches_native_sampler(minimum, legacy_config):
    params = SamplingParams() if minimum is None else SamplingParams(min_tokens=minimum)
    if legacy_config:
        legacy = asdict(params)
        # The legacy schema stores extra native kwargs directly, not in this typed-only field.
        legacy.pop("additional_kwargs")
        if minimum is None:
            legacy.pop("min_tokens")
        params = OmegaConf.create(legacy)
    native = VllmSamplingParams(**get_vllm_sampling_params(params))
    assert native.min_tokens == (1 if minimum is None else minimum)
