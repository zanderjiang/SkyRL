from enum import Enum


class InterpolationMode(str, Enum):
    """Real enum members: `transformers.image_utils` builds a PIL<->torchvision resampling map at
    import time and indexes these by name."""

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def __getattr__(name):
    def _f(*a, **k):
        raise RuntimeError(f"torchvision stub: transforms.functional.{name}")
    return _f
