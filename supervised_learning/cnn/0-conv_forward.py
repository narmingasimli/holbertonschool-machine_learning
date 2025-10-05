#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Convolve Forward"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)

    h_out = int(np.floor((h_prev + 2 * ph - kh) / sh) + 1)
    w_out = int(np.floor((w_prev + 2 * pw - kw) / sw) + 1)
    output = np.zeros((m, h_out, w_out, c_new))
    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    slice = A_prev_pad[i, vert_start:vert_end,
                                       horiz_start:horiz_end, :]

                    conv = (np.sum(slice * W[:, :, :, c])
                            + b[:, :, :, c])

                    output[i, h, w, c] = activation(conv)
    return output
