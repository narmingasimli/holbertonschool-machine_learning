#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Optimize Model"""
    optimize = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1,
                                 beta_2=beta2)
    network.compile(loss='categorical_crossentropy',
                    metrics=["accuracy"], optimizer=optimize)
