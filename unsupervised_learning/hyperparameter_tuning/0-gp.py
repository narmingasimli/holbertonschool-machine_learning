#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


class GaussianProcess:
    """Gaussian Process Class that represents a noiseless
    1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize variables"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices"""
        sqr = (X1 - X2.T) ** 2
        k = self.sigma_f ** 2 * np.exp(-sqr / (2 * self.l ** 2))
        return k
