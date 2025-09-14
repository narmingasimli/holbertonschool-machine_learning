#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffle Data"""
    i = X.shape[0]
    perm = np.random.permutation(i)
    return X[perm], Y[perm]
