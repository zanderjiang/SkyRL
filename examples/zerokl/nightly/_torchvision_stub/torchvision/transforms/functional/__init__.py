InterpolationMode = type("InterpolationMode", (), {"BICUBIC": "bicubic", "BILINEAR": "bilinear"})


def __getattr__(name):
    def _f(*a, **k):
        raise RuntimeError(f"torchvision stub: transforms.functional.{name}")
    return _f
