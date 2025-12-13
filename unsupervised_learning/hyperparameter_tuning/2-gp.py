#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


class GaussianProcess:
    """Gaussian Process Class that
    represents a noiseless 1D Gaussian process"""

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

    def predict(self, X_s):
        """Function predicts the mean and
        standard deviation of points in a Gaussian process"""
        K_s = self.kernel(X_s, self.X)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = K_s @ K_inv @ self.Y
        mu = mu.ravel()
        cov = K_ss - K_s @ K_inv @ K_s.T
        sigma = np.diag(cov)
        return mu, sigma

    def update(self, X_new, Y_new):
        """Function updates a Gaussian Process"""
        X_new = np.atleast_2d(X_new)
        Y_new = np.atleast_2d(Y_new)
        k_new = self.kernel(X_new, self.X)
        k_nn = self.kernel(X_new, X_new)
        top = np.hstack([self.K, k_new.T])
        bottom = np.hstack([k_new, k_nn])
        self.K = np.vstack([top, bottom])
        self.X = np.vstack([self.X, X_new])
        self.Y = np.vstack([self.Y, Y_new])
