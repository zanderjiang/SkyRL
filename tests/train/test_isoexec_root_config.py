"""Root model settings reach every native actor without entering TrainerConfig."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from skyrl.train.config import SkyRLTrainConfig


@pytest.mark.parametrize("key", ["trainer.enable_isoexec", "trainer.isoexec_model"])
def test_old_trainer_fields_are_rejected(key):
    with pytest.raises(ValueError, match="Invalid fields"):
        SkyRLTrainConfig.from_cli_overrides([key + "=true"])


@pytest.mark.parametrize("enabled", [False, True])
def test_root_settings_are_forwarded_to_master_and_peer_actors(monkeypatch, enabled):
    import ray

    # This actor-forwarding test uses stand-ins; keep import-time registries local.
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    from skyrl.backends.skyrl_train.workers import worker

    cfg = SkyRLTrainConfig.from_cli_overrides([f"enable_isoexec={str(enabled).lower()}"])
    cfg.isoexec_model = {"source": {"api_version": 2, "package": "isoexec.models.v2.qwen3_5"}} if enabled else None
    received = []

    class Actor:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            received.append(kwargs)
            return SimpleNamespace(
                get_master_addr_port=SimpleNamespace(remote=lambda: ("localhost", 23456)),
                init_worker_process_group=SimpleNamespace(remote=lambda: None),
                get_mesh_rank=SimpleNamespace(remote=lambda: SimpleNamespace(dp_size=2)),
            )

    monkeypatch.setattr(worker.ray, "get", lambda value: value)
    group = worker.PPORayActorGroup(
        cfg.trainer, 2, 1, cast(Any, Actor()), enable_isoexec=cfg.enable_isoexec, isoexec_model=cfg.isoexec_model
    )
    assert len(group._actor_handlers) == len(received) == 2
    assert not hasattr(cfg.trainer, "enable_isoexec")
    assert not hasattr(cfg.trainer, "isoexec_model")
    for rank, arguments in enumerate(received):
        assert arguments["rank"] == rank
        assert arguments["cfg"] is cfg.trainer
        assert arguments["enable_isoexec"] is enabled
        assert json.loads(json.dumps(arguments["isoexec_model"])) == cfg.isoexec_model
