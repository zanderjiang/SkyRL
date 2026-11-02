"""Profiler pass-through arguments must be normalized before HTTP app setup."""

import asyncio
from argparse import Namespace

import pytest


@pytest.mark.vllm
@pytest.mark.parametrize("kind", ["mapping", "typed", "disabled"])
def test_server_profiler_config_registers_real_router(monkeypatch, tmp_path, kind):
    from fastapi import FastAPI
    from vllm.config import ProfilerConfig
    from vllm.entrypoints.serve.profile.api_router import attach_router

    from skyrl.backends.skyrl_train.inference_servers import vllm_server_actor as server

    values = {"profiler": "torch", "torch_profiler_dir": str(tmp_path)}
    config = values if kind == "mapping" else ProfilerConfig(**values) if kind == "typed" else None
    args = Namespace(host="127.0.0.1", port=8000, profiler_config=config)

    class AppChecked(Exception):
        pass

    def build_app(cli_args):
        app = FastAPI()
        app.state.args = cli_args
        attach_router(app)
        paths = {getattr(route, "path", "") for route in app.routes}
        assert ("/start_profile" in paths) == (kind != "disabled")
        assert ("/stop_profile" in paths) == (kind != "disabled")
        if kind == "typed":
            assert cli_args.profiler_config is config
        raise AppChecked

    monkeypatch.setattr(server, "_seed_dp_master_port", lambda port: None)
    monkeypatch.setattr(server, "create_server_socket", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "build_app", build_app)
    with pytest.raises(AppChecked):
        asyncio.run(server._build_and_serve_vllm_server(args))
