#!/usr/bin/env python3
"""Gaussian probability density function"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the PDF of a Gaussian distribution
    Args:
        X: numpy.ndarray of shape (n, d) containing the data points
        m: numpy.ndarray of shape (d,) containing the mean
        S: numpy.ndarray of shape (d, d) containing the covariance
    Returns: P, numpy.ndarray of shape (n,) containing the PDF values
             of each data point, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    d = X.shape[1]
    if m.shape[0] != d or S.shape != (d, d):
        return None

    diff = X - m
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    expo = -0.5 * np.sum(np.matmul(diff, inv) * diff, axis=1)
    P = np.exp(expo) / np.sqrt(((2 * np.pi) ** d) * det)
    return np.maximum(P, 1e-300)
