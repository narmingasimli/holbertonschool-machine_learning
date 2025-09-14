#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Create Momentum Optimizer"""
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
