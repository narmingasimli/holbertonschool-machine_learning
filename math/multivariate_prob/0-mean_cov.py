#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def mean_cov(X):
    """Comment of Function"""
    if type(X) is not np.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = X.mean(axis=0).reshape(1, d)
    X_mean = X - mean
    cov = (X_mean.T @ X_mean) / (n - 1)
    return mean, cov
