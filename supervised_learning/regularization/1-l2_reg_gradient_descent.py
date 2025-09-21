#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """L2 Regularization Gradient Descent"""
    AL = cache['A{}'.format(L)]
    dZ = AL - Y
    m = Y.shape[1]
    for i in range(L, 0, -1):
        W = weights['W{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]
        dW = (1/m) * (dZ @ A_prev.T) + (lambtha/m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
            dZ_prev = (W.T @ dZ) * (1 - A_prev ** 2)
        weights['W{}'.format(i)] -= alpha * dW
        weights['b{}'.format(i)] -= alpha * db
        if i > 1:
            dZ = dZ_prev
