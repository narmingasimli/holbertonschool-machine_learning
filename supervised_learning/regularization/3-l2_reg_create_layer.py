#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """L2 Regularization Create Layer"""
    regularizer = tf.keras.regularizers.l2(lambtha)
    layer_weight = tf.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    output_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_initializer=layer_weight
    )(prev)
    return output_layer
