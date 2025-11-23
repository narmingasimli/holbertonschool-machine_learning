#!/usr/bin/env python3
"""Probability"""


class Poisson:
    """Poisson Class"""
    def __init__(self, data=None, lambtha=1.):
        """Initiailize Function"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """Probability Mass Function)"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e = 2.7182818285
        e_term = 1 / (e ** self.lambtha)
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        return (self.lambtha ** k) * e_term / factorial

    def cdf(self, k):
        """Cumulative Distribution Function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
