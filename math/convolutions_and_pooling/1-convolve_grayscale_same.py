#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Convolve Grayscale Same"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph_top = kh // 2
    ph_bottom = kh - 1 - ph_top
    pw_left = kw // 2
    pw_right = kw - 1 - pw_left
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
        mode='constant',
        constant_values=0
    )
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            image_patch = images_padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_patch * kernel, axis=(1, 2))
    return output
