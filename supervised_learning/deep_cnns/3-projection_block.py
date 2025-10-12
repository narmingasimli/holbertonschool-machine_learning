#!/usr/bin/env python3
"""Comment of Function"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Projection Block"""
    F11, F3, F12 = filters
    X_shortcut = A_prev
    he_normal = K.initializers.HeNormal(seed=0)

    X = K.layers.Conv2D(F11, (1, 1), strides=s, padding='valid',
                        kernel_initializer=he_normal)(A_prev)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), padding='valid',
                        kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization()(X)

    X_shortcut = K.layers.Conv2D(F12, (1, 1), strides=s,
                                 padding='valid',
                                 kernel_initializer=he_normal)(X_shortcut)
    X_shortcut = K.layers.BatchNormalization()(X_shortcut)

    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
