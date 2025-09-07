#!/usr/bin/env python3
"""Function is starting"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Inside of Function
    """
    return np.concatenate((mat1, mat2), axis=axis)
