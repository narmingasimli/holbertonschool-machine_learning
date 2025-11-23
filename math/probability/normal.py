#!/usr/bin/env python3
"""Probability"""
pi = 3.1415926536
e = 2.7182818285


class Normal:
    """Normal Class"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Initiailize Function"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            var = sum((x - self.mean) ** 2 for x in data)
            self.stddev = (var / len(data)) ** 0.5
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """Z_Score Function"""
        z_score = (x - self.mean) / self.stddev
        return z_score

    def x_value(self, z):
        """X_Value Function"""
        x_value = self.mean + z * self.stddev
        return x_value

    def pdf(self, x):
        """Probability Density Function"""
        pi = 3.1415926536
        e = 2.7182818285
        part1 = 1 / (self.stddev * (2 * pi) ** 0.5)
        part2 = e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        pdf = part1 * part2

        return pdf

    @staticmethod
    def erf(x):
        return ((2 / (pi) ** 0.5) *
                (x - (x ** 3) / 3 +
                 (x ** 5) / 10 -
                 (x ** 7) / 42 +
                 (x ** 9) / 216))

    def cdf(self, x):
        """Cumulative Distribution Function"""
        part1 = (x - self.mean) / (self.stddev * (2 ** 0.5))
        cdf = 1 / 2 * (1 + self.erf(part1))

        return cdf
