#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Adam Function"""
    optimizer = tf.keras.optimizers.Adam(alpha, beta1, beta2, epsilon)
    return optimizer
