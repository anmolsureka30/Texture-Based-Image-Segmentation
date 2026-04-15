"""Evaluation metrics and cluster-label alignment.

- ARI (vs ground truth)
- Silhouette
- Davies-Bouldin
- Hungarian matching so predicted cluster labels align with GT colours,
  making side-by-side maps visually comparable.
"""
from __future__ import annotations
import numpy as np


def align_labels(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Permute `pred` labels to best match `ref` via Hungarian assignment."""
    raise NotImplementedError


def evaluate(features: np.ndarray, pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict:
    """Return dict with keys: ari, silhouette, davies_bouldin."""
    raise NotImplementedError


def labels_to_map(labels: np.ndarray, centers: list[tuple[int, int]],
                  image_shape: tuple[int, int], window: int) -> np.ndarray:
    """Paint per-window labels back onto an image-shaped segmentation map."""
    raise NotImplementedError
