#!/usr/bin/env python3
"""Expectation Maximization"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """
    Performs the expectation maximization algorithm on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, the number of clusters
        iterations: positive integer, the maximum number of iterations
        tol: non-negative float, the tolerance of the log likelihood
        verbose: boolean, whether to print the log likelihood
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
        pi: numpy.ndarray of shape (k,) with the priors
        m: numpy.ndarray of shape (k, d) with the centroid means
        S: numpy.ndarray of shape (k, d, d) with the covariances
        g: numpy.ndarray of shape (k, n) with the posteriors
        l: the log likelihood of the model
    """
    fail = (None, None, None, None, None)
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return fail
    if not isinstance(k, int) or k <= 0:
        return fail
    if not isinstance(iterations, int) or iterations <= 0:
        return fail
    if not isinstance(tol, float) or tol < 0:
        return fail
    if not isinstance(verbose, bool):
        return fail

    pi, m, S = initialize(X, k)
    g, lik = expectation(X, pi, m, S)
    l_prev = 0
    msg = "Log Likelihood after {} iterations: {}"

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print(msg.format(i, lik.round(5)))
        pi, m, S = maximization(X, g)
        g, lik = expectation(X, pi, m, S)
        if abs(l_prev - lik) <= tol:
            break
        l_prev = lik

    if verbose:
        print(msg.format(i + 1, lik.round(5)))
    return pi, m, S, g, lik
