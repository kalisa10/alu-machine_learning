#!/usr/bin/env python3
"""Expectation step"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm of a GMM
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        pi: numpy.ndarray of shape (k,) containing the priors
        m: numpy.ndarray of shape (k, d) containing the centroid means
        S: numpy.ndarray of shape (k, d, d) containing the covariances
    Returns: g, l, or None, None on failure
        g: numpy.ndarray of shape (k, n) with the posterior probabilities
           of each data point in each cluster
        l: the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    prob = np.zeros((k, n))
    for i in range(k):
        prob[i] = pi[i] * pdf(X, m[i], S[i])
    total = np.sum(prob, axis=0)
    g = prob / total
    lik = np.sum(np.log(total))
    return g, lik
