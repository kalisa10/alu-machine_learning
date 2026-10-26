#!/usr/bin/env python3
"""K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, the number of clusters
        iterations: positive integer, the maximum number of iterations
    Returns: C, clss, or None, None on failure
        C: numpy.ndarray of shape (k, d) with the centroid means
        clss: numpy.ndarray of shape (n,) with the index of the cluster
              in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for i in range(iterations):
        prev = C.copy()
        dist = np.linalg.norm(X[:, None] - C[None], axis=2)
        clss = np.argmin(dist, axis=1)
        for j in range(k):
            idx = np.where(clss == j)[0]
            if len(idx) == 0:
                C[j] = np.random.uniform(low, high, size=(1, d))
            else:
                C[j] = X[idx].mean(axis=0)
        if np.array_equal(prev, C):
            break

    dist = np.linalg.norm(X[:, None] - C[None], axis=2)
    clss = np.argmin(dist, axis=1)
    return C, clss
