#!/usr/bin/env python3
"""Probability"""


class Exponential:
    """Exponential Class"""
    def __init__(self, data=None, lambtha=1.):
        """Initiailize Function"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """Probability Mass Function"""
        if x < 0:
            return 0
        e = 2.7182818285
        pdf = self.lambtha * (1 / (e ** (self.lambtha * x)))
        return pdf

    def cdf(self, x):
        """Cumulative Distribution Function"""
        if x < 0:
            return 0
        e = 2.7182818285
        cdf = 1 - (1 / (e ** (self.lambtha * x)))
        return cdf
