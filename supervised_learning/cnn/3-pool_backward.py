#!/usr/bin/env python3
"""Comment of Function"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Pooling Backward"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    slice = A_prev[i, vert_start:vert_end,
                                   horiz_start:horiz_end, ch]

                    if mode == 'max':
                        mask = (slice == np.max(slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, ch] += (
                                    mask * dA[i, h, w, ch])
                    elif mode == 'avg':
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, ch] += (
                                    dA[i, h, w, ch] / (kh * kw))

    return dA_prev
