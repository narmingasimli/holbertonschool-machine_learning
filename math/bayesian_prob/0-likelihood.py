#!/usr/bin/env python3
"""Bayesian Probability"""
import numpy as np


def likelihood(x, n, P):
    """Bayesian Probability    """
    if (type(n) is not int) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if (type(x) is not int) or (x < 0):
        raise ValueError("x must be an integer that is greater than or equal "
                         "to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (type(P) is not np.ndarray) or (len(P.shape) != 1):
        raise TypeError("P must be a 1D numpy.ndarray")
    for p in P:
        if not (p >= 0 and p <= 1):
            raise ValueError("All values in P must be in the range [0, 1]")

    num = np.math.factorial(n)
    den = np.math.factorial(x) * np.math.factorial(n - x)

    like = num / den * (P ** x) * ((1 - P) ** (n - x))

    return like
