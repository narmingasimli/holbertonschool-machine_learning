#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def dropout_gradient_descent(Y, weights,
                             cache, alpha,
                             keep_prob, L):
    """Dropout Gradient Descent"""
    dZ = cache.get("A{}".format(L)) - Y
    m = Y.shape[1]

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(i - 1)]
        W = weights["W{}".format(i)]

        dW = (dZ @ A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA_prev = W.T @ dZ
            D_prev = cache["D{}".format(i - 1)]
            dA_prev = dA_prev * D_prev / keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)

        weights["W{}".format(i)] -= alpha * dW
        weights["b{}".format(i)] -= alpha * db
