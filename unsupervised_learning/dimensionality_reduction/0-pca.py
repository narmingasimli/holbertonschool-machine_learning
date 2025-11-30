#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def pca(X, var=0.95):
    """Principal Component Analysis"""
    x_centered = X - np.mean(X, axis=0)

    cov = np.dot(x_centered.T, x_centered) / x_centered.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = np.sum(eigenvalues)
    cumulative_var = np.cumsum(eigenvalues) / total_var

    k = np.argmax(cumulative_var >= var) + 1

    w = eigenvectors[:, :k]
    
    X_transformed = np.dot(x_centered, w)

    return X_transformed, w
