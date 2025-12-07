#!/usr/bin/env python3
"""Agglomerative clustering"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset"""
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')
    
    clss = scipy.cluster.hierarchy.fcluster(linkage_matrix, dist,
                                             criterion='distance')
    
    plt.figure(figsize=(10, 7))
    scipy.cluster.hierarchy.dendrogram(linkage_matrix,
                                        color_threshold=dist)
    plt.show()
    
    return clss
