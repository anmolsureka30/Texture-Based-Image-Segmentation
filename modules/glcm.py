"""Gray Level Co-occurrence Matrix (GLCM) from scratch.

Core texture-description object. Counts pixel-pair co-occurrences at a
given distance and angle, optionally averaged over 4 directions for
rotation invariance. Sliding-window helper produces a GLCM per window.
"""
from __future__ import annotations
import numpy as np


def build_glcm(patch: np.ndarray, n_levels: int, distance: int = 1, angle: float = 0.0) -> np.ndarray:
    """Return the (n_levels x n_levels) normalised GLCM for one patch."""
    raise NotImplementedError


def build_glcm_all_directions(patch: np.ndarray, n_levels: int, distance: int = 1) -> np.ndarray:
    """Average GLCM over angles {0, 45, 90, 135} for rotation invariance."""
    raise NotImplementedError


def sliding_glcms(image: np.ndarray, n_levels: int, window: int = 32, step: int = 16,
                  distance: int = 1) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Compute rotation-invariant GLCM at every sliding-window position.

    Returns (glcms, centers) where glcms has shape (n_windows, n_levels, n_levels)
    and centers lists the (row, col) pixel at the centre of each window.
    """
    raise NotImplementedError
