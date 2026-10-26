#!/usr/bin/env python3
"""Bayesian Information Criterion"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters using the BIC
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, the minimum number of clusters to check
        kmax: positive integer, the maximum number of clusters to check
        iterations: positive integer, the maximum number of iterations
        tol: non-negative float, the tolerance of the EM algorithm
        verbose: boolean, whether the EM algorithm prints information
    Returns: best_k, best_result, l, b, or None, None, None, None
        best_k: the best value of k based on its BIC
        best_result: tuple (pi, m, S) with the best parameters
        l: numpy.ndarray of shape (kmax - kmin + 1) with the log
           likelihood of each cluster size
        b: numpy.ndarray of shape (kmax - kmin + 1) with the BIC value
           of each cluster size
    """
    fail = (None, None, None, None)
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return fail
    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin <= 0:
        return fail
    if not isinstance(kmax, int) or kmax <= 0 or kmax <= kmin:
        return fail
    if not isinstance(iterations, int) or iterations <= 0:
        return fail
    if not isinstance(tol, float) or tol < 0:
        return fail
    if not isinstance(verbose, bool):
        return fail

    ks = list(range(kmin, kmax + 1))
    results = []
    lik = np.zeros(len(ks))
    b = np.zeros(len(ks))
    for idx, k in enumerate(ks):
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        results.append((pi, m, S))
        lik[idx] = ll
        p = k * d + k * d * (d + 1) / 2 + k - 1
        b[idx] = p * np.log(n) - 2 * ll

    best = np.argmin(b)
    return ks[best], results[best], lik, b
