#!/usr/bin/env python3

def poly_integral(poly, C=0):
    """That calculates the integral of a polynomial:"""

    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    if not isinstance(C, int):
        return None

    integral_poly = [C]

    for i, coeff in enumerate(poly):
        new_coeff = coeff / (i + 1)
        if new_coeff == int(new_coeff):
            integral_poly.append(int(new_coeff))
        else:
            integral_poly.append(new_coeff)

    while len(integral_poly) > 1 and integral_poly[-1] == 0:
        integral_poly.pop()
    
    return integral_poly
