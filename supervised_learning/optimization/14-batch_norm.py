#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Batch Normaliztion Layer"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    mean, variance = tf.nn.moments(dense, axes=[0])
    epsilon = 1e-7

    Z_norm = (dense - mean) / tf.sqrt(variance + epsilon)
    Z_t = gamma * Z_norm + beta

    return activation(Z_t)
