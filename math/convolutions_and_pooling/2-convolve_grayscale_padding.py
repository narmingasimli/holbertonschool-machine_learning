#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Convolve Grayscale Same"""
    kh, kw = kernel.shape
    ph, pw = padding
    
    if ph:
        pad_h_l = ph
        pad_h_r = ph
    else:
        pad_h_l = 0
        pad_h_r = 0

    if pw:
        pad_w_l = pw
        pad_w_r = pw
    else:
        pad_w_l = 0
        pad_w_r = 0

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad_h_l, pad_h_r), (pad_w_l, pad_w_r)),
        mode='constant'
    )

    m, h, w = padded_images.shape
    conv_h = h - kh + 1
    conv_w = w - kw + 1
    output = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            region = padded_images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
