#!/usr/bin/env python3
"""Classification algorithm using DNN class for multiclass classification."""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Deep Neural Network class for multiclass classification."""

    def __init__(self, nx, layers, activation='sig'):
        """Construct the deep neural network object."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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
        """Get the value of length of layers."""
        return self.__L

    @property
    def cache(self):
        """Get the value of cache."""
        return self.__cache

    @property
    def weights(self):
        """Get the value of the weights."""
        return self.__weights

    @property
    def activation(self):
        """Get the value of the activation function."""
        return self.__activation

    def forward_prop(self, X):
        """Calculate forward propagation of the neural network."""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(i)]
            A = self.__cache['A{}'.format(i-1)]
            b = self.__weights['b{}'.format(i)]
            z = np.dot(W, A) + b

            if i == self.__L:
                exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
                self.__cache['A{}'.format(i)] = \
                    exp_z / np.sum(exp_z, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    self.__cache['A{}'.format(i)] = 1 / (1 + np.exp(-z))
                elif self.__activation == 'tanh':
                    self.__cache['A{}'.format(i)] = np.tanh(z)

        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression."""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions."""
        self.forward_prop(X)
        A = self.__cache['A{}'.format(self.__L)]
        cost = self.cost(Y, A)
        predictions = np.zeros_like(A)
        max_indices = np.argmax(A, axis=0)
        predictions[max_indices, np.arange(A.shape[1])] = 1

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculate one pass of gradient descent on the neural network."""
        m = Y.shape[1]

        for i in range(self.L, 0, -1):

            A_prev = cache["A" + str(i - 1)]
            A = cache["A" + str(i)]
            W = self.weights["W" + str(i)]

            if i == self.L:
                dz = A - Y
            else:
                if self.activation == 'sig':
                    dz = da * (A * (1 - A))
                elif self.activation == 'tanh':
                    dz = da * (1 - A**2)

            db = dz.mean(axis=1, keepdims=True)
            dw = np.matmul(dz, A_prev.T) / m
            da = np.matmul(W.T, dz)
            self.weights['W' + str(i)] -= (alpha * dw)
            self.weights['b' + str(i)] -= (alpha * db)

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """ Train the deep neural network."""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        costs = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if verbose and i % step == 0:

                cost = self.cost(Y, self.cache["A"+str(self.L)])
                costs.append(cost)
                print('Cost after {} iterations: {}'.format(i, cost))
        if graph:
            plt.plot(np.arange(0, iterations, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Save the instance object to a file in pickle format."""
        try:
            if not filename.endswith(".pkl"):
                filename += ".pkl"
            with open(filename, "wb") as file:
                pickle.dump(self, file)
        except Exception:
            return None

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object."""
        try:
            with open(filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
