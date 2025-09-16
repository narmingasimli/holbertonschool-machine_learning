#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create Confusion Matrix"""
    m, classes = labels.shape
    conf_mat = np.zeros((classes, classes))
    for i in range(m):
        real = np.argmax(labels[i])
        pred = np.argmax(logits[i])
        conf_mat[real][pred] += 1
    return conf_mat
