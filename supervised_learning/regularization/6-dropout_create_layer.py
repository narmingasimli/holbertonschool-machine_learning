#!/usr/bin/env python3
"""Create a Layer with Dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation,
                         keep_prob, training=True):
    """This function creates a layer
     of a neural network using dropout"""
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )
    layer = tf.keras.layers.Dense(
        units=n, activation=activation,
        kernel_initializer=initializer,
    )(prev)

    if training:
        dropout = tf.nn.dropout(layer, rate=1 - keep_prob)
    return dropout
