#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""


import numpy as np


def one_hot_decode(one_hot):
    """One-Hot Decode Function"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    try:
        one_hot_decode = np.argmax(one_hot, axis=0)
        return one_hot_decode
    except Exception:
        return None
