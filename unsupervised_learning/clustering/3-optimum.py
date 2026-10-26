#!/usr/bin/env python3
"""Optimize k"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests the optimum number of clusters by variance
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, the minimum number of clusters to check
        kmax: positive integer, the maximum number of clusters to check
        iterations: positive integer, the maximum number of iterations
    Returns: results, d_vars, or None, None on failure
        results: list of the outputs of K-means per cluster size
        d_vars: list of the difference in variance from the smallest
                cluster size per cluster size
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax <= kmin:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        var = variance(X, C)
        if k == kmin:
            first = var
        d_vars.append(first - var)
    return results, d_vars
