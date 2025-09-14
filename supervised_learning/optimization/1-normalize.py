#!/usr/bin/env python3
"""Normalization Function"""
import numpy as np


def normalize(X, m, s):
    """Normalize Function"""
    return (X - m) / s
