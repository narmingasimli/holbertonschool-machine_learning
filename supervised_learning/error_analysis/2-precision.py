#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def precision(confusion):
    """ Precision"""
    # Precision = TP / (TP + FP)
    TP = np.diagonal(confusion)
    TP_FP = np.sum(confusion, axis=0)
    return TP / TP_FP
