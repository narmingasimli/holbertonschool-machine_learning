#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test Model"""
    if verbose:
        verbose = 1
    else:
        verbose = 0
    test = network.evaluate(data, labels, verbose=verbose)
    return test
