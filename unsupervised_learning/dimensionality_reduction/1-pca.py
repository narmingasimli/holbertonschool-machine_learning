#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def pca(X, ndim):
    """Principal Component Analysis"""
    X_centered = X - np.mean(X, axis=0)
    
    cov = np.dot(X_centered.T, X_centered) / X.shape[0]
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    
    W = eigenvectors[:, :ndim]
    
    T = np.dot(X_centered, W)
    
    return T
