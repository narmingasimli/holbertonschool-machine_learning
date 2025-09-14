#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Batch Normaliztion"""
    mu = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_ = (Z - mu) / (np.sqrt(var + epsilon))
    Z_norm = gamma * Z_ + beta
    return Z_norm
