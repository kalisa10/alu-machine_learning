#!/usr/bin/env python3
"""K-means with sklearn"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: the number of clusters
    Returns: C, clss
        C: numpy.ndarray of shape (k, d) with the centroid means
        clss: numpy.ndarray of shape (n,) with the cluster index of
              each data point
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
