#!/usr/bin/env python3
"""Comment of Function"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def identity_block(A_prev, filters):
    """Identity Block"""
    F11, F3, F12 = filters
    A_shortcut = A_prev
    init = K.initializers.HeNormal(seed=0)

    X = K.layers.Conv2D(F11, (1, 1), strides=(1, 1), padding='valid',
                        kernel_initializer=init)(A_prev)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), strides=(1, 1), padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), strides=(1, 1), padding='valid',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization()(X)

    X = K.layers.Add()([X, A_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
