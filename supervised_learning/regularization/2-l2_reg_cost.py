#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """L2 Regularization Cost"""
    total_costs = [cost + loss for loss in model.losses]
    return tf.stack(total_costs)
