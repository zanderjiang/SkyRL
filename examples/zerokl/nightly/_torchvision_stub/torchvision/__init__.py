"""Minimal torchvision import stub (zerokl venv has no torchvision).
vLLM's Qwen3.5 module imports the VL chain unconditionally; only import-time attribute
access is needed for a text-only run."""
__version__ = "0.0.0-zerokl-stub"
from . import transforms, io, ops  # noqa: F401,E402


def _make(name):
    raise RuntimeError(f"torchvision stub: {name} is not available")
