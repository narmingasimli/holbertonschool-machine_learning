#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """L2 Regularization"""
    weight_power = 0
    for key in weights:
        if key.startswith('W'):
            weight_power += np.sum(weights[key] ** 2)
    loss_regul = (lambtha / (2 * m)) * weight_power
    total_loss = cost + loss_regul
    return total_loss
