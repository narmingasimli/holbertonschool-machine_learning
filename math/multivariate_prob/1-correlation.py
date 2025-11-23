#!/usr/bin/env python3
"""
Comment of Function
"""
import numpy as np


def correlation(C):
    """
    Comment of Function
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    diag = np.diag(C).reshape(1, -1)
    stddev = np.sqrt(diag)

    corr = C / (stddev * stddev.T)
    return corr
