#!/usr/bin/env python3


"""Classification"""


import numpy as np


class Neuron:
    """Neuron Class"""

    def __init__(self, nx):
        """Init Function"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter Method"""
        return self.__W

    @property
    def b(self):
        """Getter Method"""
        return self.__b

    @property
    def A(self):
        """Getter Method"""
        return self.__A

    def forward_prop(self, X):
        """Forward Propagation"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
