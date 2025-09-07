#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Predict Function"""
    if verbose:
        verbose = 1
    else:
        verbose = 0
    predict = network.predict(data, verbose=verbose)
    return predict
