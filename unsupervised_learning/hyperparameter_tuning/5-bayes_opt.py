#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Class that performs
    Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initialize variables"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Function that calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            improve = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            improve = mu - best - self.xsi

        Z = np.zeros_like(mu)
        Z[sigma != 0] = improve[sigma != 0] / sigma[sigma != 0]

        EI = np.zeros_like(mu)
        nonzero = sigma != 0
        EI[nonzero] = improve[nonzero] * norm.cdf(Z[nonzero]) + \
            sigma[nonzero] * norm.pdf(Z[nonzero])

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Function that optimizes the black-box function"""
        x_searched = []
        for _ in range(iterations):
            x, _ = self.acquisition()
            y = self.f(x)
            if x in x_searched:
                break
            x_searched.append(x)
            self.gp.update(x, y)
        self.gp.X = self.gp.X[:-1]
        if self.minimize:
            y_opt = np.min(self.gp.Y, keepdims=True)
            return self.gp.X[np.argmin(self.gp.Y)], y_opt[0]
        else:
            y_opt = np.max(self.gp.Y, keepdims=True)
            return self.gp.X[np.argmax(self.gp.Y)], y_opt[0]
