#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n:
        return None, None, None
    column_sums = np.sum(g, axis=0)
    if not np.allclose(column_sums, 1):
        return None, None, None
    total_prob = np.sum(g, axis=1)
    pi = total_prob / n
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        m[i] = np.sum(g[i, :, np.newaxis] * X, axis=0) / total_prob[i]
        diff = X - m[i]
        S[i] = (g[i, :, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] *
                diff[:, np.newaxis, :]).sum(axis=0) / total_prob[i]
    return pi, m, S
