#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def save_model(network, filename):
    """Save Model Function"""
    network.save(filename)


def load_model(filename):
    """Load Model Function"""
    model = K.models.load_model(filename)
    return model
