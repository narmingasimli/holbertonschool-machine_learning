#!/usr/bin/env python3
"""Early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Early stopping"""
    if cost < opt_cost - threshold:
        opt_cost = cost
        count = 0

    else:
        count += 1

    stop = count >= patience
    return stop, count
