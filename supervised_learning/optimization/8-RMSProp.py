#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """RMSProbFunction in TenserFlow"""
    optimizer = tf.keras.optimizers.RMSprop(alpha, beta2, epsilon)
    return optimizer
