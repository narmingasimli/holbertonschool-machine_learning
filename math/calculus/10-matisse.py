#!/usr/bin/env python3
"""That calculates the derivative of a polynomial:"""


def poly_derivative(poly):
    """Return number"""
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None

    if len(poly) == 1:
        return [0]

    derivative = [i * poly[i] for i in range(1, len(poly))]

    return derivative if derivative else [0]
