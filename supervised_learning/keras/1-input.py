#!/usr/bin/env python3
"""Input Class"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build Model Function"""
    regularizer = K.regularizers.l2(lambtha)
    drop = 1 - keep_prob
    input_model = K.Input(shape=(nx,))
    x = input_model

    for i in range(len(layers)):
        x = K.layers.Dense(units=layers[i],
                           activation=activations[i],
                           kernel_regularizer=regularizer)(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=drop)(x)

    model = K.models.Model(inputs=input_model, outputs=x)
    return model
