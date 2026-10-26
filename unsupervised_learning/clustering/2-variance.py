#!/usr/bin/env python3
"""Variance"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance of a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        C: numpy.ndarray of shape (k, d) containing the centroid means
    Returns: var, the total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    dist = np.linalg.norm(X[:, None] - C[None], axis=2)
    return np.sum(np.min(dist, axis=1) ** 2)
