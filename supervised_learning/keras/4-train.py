#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Train Model"""
    History = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          verbose=verbose, shuffle=shuffle)
    return History
