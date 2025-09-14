#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def moving_average(data, beta):
    """Moving Average"""
    v_t = 0
    mov_avg = []
    for i in range(len(data)):
        v_t = (beta * v_t) + ((1 - beta) * data[i])
        bias_correction = 1 - (beta ** (i + 1))
        mov_avg.append(v_t / bias_correction)
    return mov_avg
