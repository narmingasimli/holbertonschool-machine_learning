#!/usr/bin/env python3
"""Probability"""


class Binomial:
    """Binomial Class"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initiailize Function"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1 - (var / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est
            self.n = n_est
            self.p = p_est
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """Probability Mass Function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        n_fact = 1
        for i in range(1, self.n + 1):
            n_fact *= i

        k_fact = 1
        for i in range(1, k + 1):
            k_fact *= i

        nk_fact = 1
        for i in range(1, self.n - k + 1):
            nk_fact *= i

        comb = n_fact / (k_fact * nk_fact)

        pmf = comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return pmf

    def cdf(self, k):
        """Cumulative Distribution Function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        cdf = 0
        for i in range(0, k + 1):
            cdf += self.pmf(i)
        return cdf
