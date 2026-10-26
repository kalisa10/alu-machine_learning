#!/usr/bin/env python3
"""Gaussian Mixture Model with sklearn"""

import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: the number of clusters
    Returns: pi, m, S, clss, bic
        pi: numpy.ndarray of shape (k,) with the cluster priors
        m: numpy.ndarray of shape (k, d) with the centroid means
        S: numpy.ndarray of shape (k, d, d) with the covariances
        clss: numpy.ndarray of shape (n,) with the cluster index of
              each data point
        bic: the BIC value of the model
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)
    return pi, m, S, clss, bic
