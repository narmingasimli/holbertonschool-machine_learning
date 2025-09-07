#!/usr/bin/env python3


"""Classification"""


import numpy as np


class NeuralNetwork:
    """Neural Network Class"""
    def __init__(self, nx, nodes):
        """Init Function"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter Method"""
        return self.__W1

    @property
    def b1(self):
        """Getter Method"""
        return self.__b1

    @property
    def A1(self):
        """Getter Method"""
        return self.__A1

    @property
    def W2(self):
        """Getter Method"""
        return self.__W2

    @property
    def b2(self):
        """Getter Method"""
        return self.__b2

    @property
    def A2(self):
        """Getter Method"""
        return self.__A2

    def forward_prop(self, X):
        """Forward Propagation"""
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Cost Function"""
        m = Y.shape[1]
        J1 = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return J1

    def evaluate(self, X, Y):
        """Evaluate Function"""
        A1, A2 = self.forward_prop(X)
        pred = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return pred, cost
