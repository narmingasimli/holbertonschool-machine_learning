#!/usr/bin/env python3
"""Clustering module"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance
    Args:
        X is a numpy.ndarray of shape (n, d)
        kmin is a positive integer containing the minimum number of clusters
        kmax is a positive integer containing the maximum number of clusters"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if isinstance(kmax, int) and kmax <= kmin:
        return None, None

    if kmax is None:
        max_cluster = X.shape[0]
    else:
        max_cluster = kmax

    results = []
    d_vars = []

    C, clss = kmeans(X, kmin, iterations)

    results.append((C, clss))

    base_var = variance(X, C)

    d_vars = [0.0]

    k = kmin + 1

    while k < max_cluster + 1:
        C, clss = kmeans(X, k, iterations)

        current_var = variance(X, C)

        results.append((C, clss))

        d_vars.append(base_var - current_var)

        k += 1

    return results, d_vars
