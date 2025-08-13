#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    if axis == 0:
        if not all(len(row) == len(mat1[0]) for row in mat1) or \
           not all(len(row) == len(mat2[0]) for row in mat2) or \
           len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    
    elif axis == 1:
        if len(mat1) != len(mat2) or \
           any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    
    return None
