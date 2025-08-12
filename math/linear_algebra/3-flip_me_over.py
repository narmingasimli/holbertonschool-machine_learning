#!/usr/bin/env python3
"""   A function def matrix_transpose(matrix):   """


def matrix_transpose(matrix):
    """   Return code range in len   """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))];
