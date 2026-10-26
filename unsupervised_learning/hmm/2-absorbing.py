#!/usr/bin/env python3
"""Absorbing Markov chain"""

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing
    Args:
        P: numpy.ndarray of shape (n, n) representing the transition matrix
           P[i, j] is the probability of transitioning from state i to j
    Returns: True if it is absorbing, False if not, None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n = P.shape[0]
    if P.shape[1] != n:
        return None

    stay = np.diagonal(P) == 1
    if not stay.any():
        return False
    if stay.all():
        return True

    # reach[i, j] > 0 when state j can be reached from state i
    step = np.eye(n) + (P > 0)
    reach = np.linalg.matrix_power(step, n) > 0
    return bool(reach[:, stay].any(axis=1).all())
