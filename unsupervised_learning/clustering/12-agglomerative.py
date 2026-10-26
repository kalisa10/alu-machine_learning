#!/usr/bin/env python3
"""Agglomerative clustering"""

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset with Ward linkage
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: the maximum cophenetic distance of all clusters
    Returns: clss, numpy.ndarray of shape (n,) with the cluster index
             of each data point
    """
    links = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(
        links, t=dist, criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(links, color_threshold=dist)
    plt.show()
    return clss
