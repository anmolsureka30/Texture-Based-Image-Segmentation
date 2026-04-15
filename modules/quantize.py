"""Multi-level Otsu gray-level quantization (from scratch).

Reduces a uint8 image (256 levels) to `n_levels` quantized levels by
searching thresholds that maximise between-class variance.
"""
from __future__ import annotations
import numpy as np


def otsu_multilevel(image: np.ndarray, n_levels: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Return (quantized_image, thresholds).

    quantized_image: same shape as `image`, values in {0, ..., n_levels-1}.
    thresholds: length n_levels-1 array of the chosen threshold values.
    """
    raise NotImplementedError
