#!/usr/bin/env python3
"""Initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means
        n is the number of data points
        d is the number of dimensions for each data point"""
    if type(k) is not int or k <= 0:
        return None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    n, d = X.shape
    clusters = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                                 size=(k, d))
    return clusters
