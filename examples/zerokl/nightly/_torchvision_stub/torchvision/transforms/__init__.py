from .v2 import _Any  # noqa: F401
from . import v2, functional  # noqa: F401,E402


def __getattr__(name):
    return _Any
