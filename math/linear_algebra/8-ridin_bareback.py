#!/usr/bin/env python3

def mat_mul(mat1, mat2):
    if not mat1 or not mat2:
        return None
        
    cols1 = len(mat1[0])
    rows2 = len(mat2)

    if cols1 != rows2:
        return None

    result_matrix = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(cols1):
                result_matrix[i][j] += mat1[i][k] * mat2[k][j]

    return result_matrix
