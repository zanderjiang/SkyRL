class _Any:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        raise RuntimeError("torchvision stub: transform called")


def __getattr__(name):
    return _Any
