#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update Variables Momentum"""
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
