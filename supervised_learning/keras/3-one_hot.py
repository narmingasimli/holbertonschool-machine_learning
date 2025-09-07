#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """One-Hot Function"""
    one_hot = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot
