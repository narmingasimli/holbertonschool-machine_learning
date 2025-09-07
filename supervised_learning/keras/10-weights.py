#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save Model Function"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Load Model Function"""
    network.load_weights(filename)
