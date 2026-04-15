"""SVD-based texture features: top-k singular values of the GLCM."""
from __future__ import annotations
import numpy as np


def svd_features(glcm: np.ndarray, k: int = 8) -> np.ndarray:
    """Return the top-k singular values of the GLCM as a length-k vector."""
    raise NotImplementedError
