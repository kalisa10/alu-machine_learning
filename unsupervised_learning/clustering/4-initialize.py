#!/usr/bin/env python3
"""Initialize GMM"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables of a Gaussian Mixture Model
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, the number of clusters
    Returns: pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,) with the priors, evenly set
        m: numpy.ndarray of shape (k, d) with the centroid means
        S: numpy.ndarray of shape (k, d, d) with the covariance
           matrices, set to identity matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    d = X.shape[1]
    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1, 1))
    return pi, m, S
