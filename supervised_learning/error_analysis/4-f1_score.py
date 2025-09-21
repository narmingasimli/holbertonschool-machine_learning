#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def f1_score(confusion):
    """F1 Score"""
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    precisions = precision(confusion)
    sensitivities = sensitivity(confusion)
    f1 = np.zeros_like(precisions)

    for i in range(len(precisions)):
        if precisions[i] + sensitivities[i] == 0:
            f1[i] = 0
        else:
            f1[i] = (2 * precisions[i] * sensitivities[i] /
                     (precisions[i] + sensitivities[i]))

    return f1
