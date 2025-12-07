
#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d):
        return None, None
    if S.shape != (k, d, d):
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    pdf = __import__('5-pdf').pdf

    g = np.zeros((k, n))

    for i in range(k):
        likelihood = pdf(X, m[i], S[i])
        if likelihood is None:
            return None, None
        g[i] = pi[i] * likelihood

    total = np.sum(g, axis=0)

    if np.any(total == 0):
        return None, None

    g = g / total

    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood

