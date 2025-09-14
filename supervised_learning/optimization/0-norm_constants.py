#!/usr/bin/env python3
"""Normalization Function"""
import numpy as np


def normalization_constants(X):
    """Normalization Constants Function"""
    M = np.mean(X, axis=0)
    S = np.std(X, axis=0)
    return M, S
