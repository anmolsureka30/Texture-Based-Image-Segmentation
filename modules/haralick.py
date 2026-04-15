"""Haralick (1973) textural features from a GLCM — all 13, from scratch.

1. Angular Second Moment (Energy)
2. Contrast
3. Correlation
4. Sum of Squares (Variance)
5. Inverse Difference Moment (Homogeneity)
6. Sum Average
7. Sum Variance
8. Sum Entropy
9. Entropy
10. Difference Variance
11. Difference Entropy
12. Information Measure of Correlation 1
13. Information Measure of Correlation 2
"""
from __future__ import annotations
import numpy as np


def haralick_features(glcm: np.ndarray) -> np.ndarray:
    """Return a length-13 feature vector from a normalised GLCM."""
    raise NotImplementedError
