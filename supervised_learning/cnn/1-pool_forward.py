#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Convolve Forward"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1
    output = np.zeros((m, h_out, w_out, c_prev))
    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_prev):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    hor_start = w * sw
                    hor_end = hor_start + kw
                    patch = A_prev[i, vert_start:vert_end,
                                   hor_start:hor_end, c]
                    if mode == "max":
                        output[i, h, w, c] = np.max(patch)
                    if mode == "avg":
                        output[i, h, w, c] = np.mean(patch)
    return output
