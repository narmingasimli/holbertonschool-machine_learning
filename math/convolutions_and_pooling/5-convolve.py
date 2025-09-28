#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Convolve Channels"""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph = 0
        pw = 0
    elif isinstance(padding, tuple):
        ph, pw = padding

    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            image_patch = images_padded[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw
            ]
            for k in range(nc):
                kernel = kernels[:, :, :, k]
                output[:, i, j, k] = np.sum(image_patch * kernel,
                                            axis=(1, 2, 3))

    return output
