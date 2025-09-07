#!/usr/bin/env python3
"""Sequential Class"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build Model Function"""
    model = k.Sequential()
    regularizer = k.regularizers.L2(lambtha)
    drop = 1 - keep_prob

    for i in range(len(layers)):
        if i == 0:
            model.add(k.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer,
                input_shape=(nx,)))
        else:
            model.add(k.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer))

        if i < len(layers) - 1:
            model.add(k.layers.Dropout(rate=drop))
    return model
