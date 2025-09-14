#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Learning Rate Decay"""
    a = alpha / (1 + decay_rate * (global_step // decay_step))
    return a
