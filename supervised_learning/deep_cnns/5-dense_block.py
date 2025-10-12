#!/usr/bin/env python3
"""Comment of Function"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Dense Block"""
    he_norm = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        out = K.layers.BatchNormalization()(X)
        out = K.layers.ReLU()(out)
        out = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                              kernel_initializer=he_norm)(out)

        out = K.layers.BatchNormalization()(out)
        out = K.layers.ReLU()(out)
        out = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                              kernel_initializer=he_norm)(out)

        X = K.layers.Concatenate()([X, out])
        nb_filters += growth_rate

    return X, nb_filters
