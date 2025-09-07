#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class."""

    def __init__(self, nx, layers):
        """Init Function"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
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
        """Getter for number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation for the deep neural network"""
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            A_prev = self.__cache[f"A{layer-1}"]

            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # sigmoid activation

            self.__cache[f"A{layer}"] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        # Perform forward propagation
        A, _ = self.forward_prop(X)

        # Compute cost
        cost = self.cost(Y, A)

        # Convert activated outputs to binary predictions
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent Function"""
        m = Y.shape[1]
        AL = cache['A{}'.format(self.__L)]
        dZl = AL - Y
        for i in range(self.__L, 0, -1):
            Al = cache['A{}'.format(i - 1)]
            dwl = (dZl @ Al.T) / m
            dbl = (np.sum(dZl, axis=1, keepdims=True)) / m

            Al_prev = cache['A{}'.format(i - 1)]
            Wl = self.__weights['W{}'.format(i)]
            if i > 1:
                dZl = (Wl.T @ dZl) * (Al_prev * (1 - Al_prev))
            self.__weights['W{}'.format(i)] -= alpha * dwl
            self.__weights['b{}'.format(i)] -= alpha * dbl

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """train"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
        return self.evaluate(X, Y)
