"""K-Means clustering from scratch + feature standardisation."""
from __future__ import annotations
import numpy as np


def standardize(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per feature."""
    raise NotImplementedError


def kmeans(X: np.ndarray, k: int, n_restarts: int = 10, max_iter: int = 300,
           tol: float = 1e-4, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (labels, centroids, inertia). Keeps best of `n_restarts` runs."""
    raise NotImplementedError
