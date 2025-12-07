#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    prev_log_likelihood = 0

    for i in range(iterations):
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == 0):
            print(f"Log Likelihood after {i} iterations: "
                  f"{log_likelihood:.5f}")

        if i > 0 and abs(log_likelihood - prev_log_likelihood) <= tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: "
                      f"{log_likelihood:.5f}")
            break

        prev_log_likelihood = log_likelihood

        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

    else:
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        if verbose:
            print(f"Log Likelihood after {iterations} iterations: "
                  f"{log_likelihood:.5f}")

    return pi, m, S, g, log_likelihood
