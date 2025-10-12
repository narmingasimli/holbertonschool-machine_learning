#!/usr/bin/env python3
"""Comment of Function"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Transition Layer"""
    he_normal = K.initializers.HeNormal(seed=0)

    filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(filters, (1, 1),
                        padding='same',
                        kernel_initializer=he_normal)(X)

    X = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                  padding='same')(X)

    return X, filters
