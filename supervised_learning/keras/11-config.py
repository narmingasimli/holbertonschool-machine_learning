#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def save_config(network, filename):
    """Save Model Function"""
    with open(filename, "w") as f:
        f.write(network.to_json())


def load_config(filename):
    """Load Model Function"""
    with open(filename, "r") as f:
        json_config = f.read()
        return K.models.model_from_json(json_config)
