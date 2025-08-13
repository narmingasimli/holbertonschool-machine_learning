#!/usr/bin/env python3

def mat_mul(mat1, mat2):
    # Check if matrices are valid and can be multiplied
    if not mat1 or not mat2:
        return None
    if len(mat1[0]) != len(mat2):
        return None

    # Get dimensions
    rows1 = len(mat1)
    cols1 = len(mat1[0])
    cols2 = len(mat2[0])

    # Initialize result matrix with zeros
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    # Perform multiplication
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
