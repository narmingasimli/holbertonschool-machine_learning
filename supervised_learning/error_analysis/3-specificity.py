#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def specificity(confusion):
    """Specificity"""
    classes = confusion.shape[0]
    spec = np.zeros(classes)

    for i in range(classes):
        total = np.sum(confusion)
        TP = confusion[i][i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i, :]) - TP
        TN = total - (TP + FP + FN)
        if TN + FP == 0:
            spec[i] = 0
        else:
            spec[i] = TN / (TN + FP)
    return spec
