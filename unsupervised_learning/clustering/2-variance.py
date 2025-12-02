#!/usr/bin/env python3
"""Clustering module"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    Args:
        X is a numpy.ndarray of shape (n, d)
        C is a numpy.ndarray of shape (k, d)
    Returns:
        var, or None on failure
            var is the total variance
    """
    try:
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        closet_centroid = np.argmin(distances, axis=1)

        var = np.sum((X - C[closet_centroid]) ** 2)

        return var
    except Exception:
        return None
