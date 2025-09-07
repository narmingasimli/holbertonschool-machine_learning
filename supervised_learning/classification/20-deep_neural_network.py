#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class."""

    def __init__(self, nx, layers):
        """Construct the deep neural network object."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            nodes = layers[i]
            prev_nodes = nx if i == 0 else layers[i - 1]

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.__weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

    @property
    def L(self):
        """Getter Method"""
        return self.__L

    @property
    def cache(self):
        """Getter Method"""
        return self.__cache

    @property
    def weights(self):
        """Getter Method"""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(i)]
            A = self.__cache['A{}'.format(i - 1)]
            b = self.__weights['b{}'.format(i)]
            z = np.dot(W, A) + b
            self.__cache['A{}'.format(i)] = 1 / (1 + np.exp(-z))

        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Cost Function"""
        m = Y.shape[1]
        J = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return J

    def evaluate(self, X, Y):
        """Evaluate"""
        self.forward_prop(X)
        e = self.__cache['A{}'.format(self.__L)]
        cost = self.cost(Y, e)
        labels = np.where(e >= 0.5, 1, 0)
        return labels, cost
