#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def pdf(X, m, S):
    """Calculate the probability density function of a Gaussian distribution"""
    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray)\
            or not isinstance(S, np.ndarray):
        return None
    if X.ndim != 2:
        return None
    n, d = X.shape
    if m.shape != (d,) or S.shape != (d, d):
        return None
    try:
        S_inv = np.linalg.inv(S)
        det_S = np.linalg.det(S)
    except np.linalg.LinAlgError:
        return None
    constant = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_S))
    diff = X - m
    exponent = np.sum(diff @ S_inv * diff, axis=1)
    P = constant * np.exp(-0.5 * exponent)
    P = np.maximum(P, 1e-300)
    return P
