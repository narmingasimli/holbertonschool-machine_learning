#!/usr/bin/env python3
"""Comment of Function"""
from tensorflow import keras as K
def inception_block(A_prev, filters):
    """Inception Block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    f1_conv = K.layers.Conv2D(F1, (1, 1), padding='same',
                              activation='relu')(A_prev)
    f3r_conv_reduce = K.layers.Conv2D(F3R, (1, 1), padding='same',
                                      activation='relu')(A_prev)
    f3_conv = K.layers.Conv2D(F3, (3, 3), padding='same',
                              activation='relu')(f3r_conv_reduce)
    f5r_conv_reduce = K.layers.Conv2D(F5R, (1, 1), padding='same',
                                      activation='relu')(A_prev)
    f5_conv = K.layers.Conv2D(F5, (5, 5), padding='same',
                              activation='relu')(f5r_conv_reduce)
    max_pooling = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                        padding='same')(A_prev)
    conv_after_pool = K.layers.Conv2D(FPP, (1, 1),
                                      padding='same',
                                      activation='relu')(max_pooling)
    output = K.layers.Concatenate(axis=-1)([f1_conv, f3_conv,
                                            f5_conv, conv_after_pool])
    return output
