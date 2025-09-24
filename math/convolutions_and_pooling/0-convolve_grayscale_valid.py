#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Convolve Grayscale Valid"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    new_h = h - kh + 1
    new_w = w - kw + 1
    output = np.zeros((m, new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            image_patch = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_patch * kernel, axis=(1, 2))
    return output
