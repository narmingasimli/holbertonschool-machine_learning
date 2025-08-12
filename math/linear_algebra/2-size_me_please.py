#!/usr/bin/env python3
""" """
def matrix_shape(matrix):
    shape = []
    current_dimension = matrix
    while isinstance(current_dimension, list):
        shape.append(len(current_dimension))
        if len(current_dimension) ==  0:
            current_dimension = current_dimension[0]
        else:
            break
    return shape;
