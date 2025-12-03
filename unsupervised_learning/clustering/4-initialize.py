#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    n, d = X.shape
    centroids, clss = kmeans(X, k, iterations=1000)
    pi = np.ones(k) / k
    m = centroids
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))

    return pi, m, S
