#!/usr/bin/env python3


"""Classification"""


import numpy as np
import matplotlib.pyplot as plt


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

    def cost(self, Y, A):
        """Cost Function"""
        m = Y.shape[1]
        J = -1 / m * np.sum((Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        return J

    def evaluate(self, X, Y):
        """Evaluate Prediction"""
        A = self.forward_prop(X)
        pred = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Gradient Descent"""
        m = Y.shape[1]
        dZ = A - Y
        dW = 1 / m * np.dot(dZ, X.T)
        db = (1/m) * np.sum(dZ)
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Train The Model"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        steps = []
        for iteration in range(iterations):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)
            self.gradient_descent(X, Y, A, alpha)
            if iteration % step == 0 or iteration == iterations - 1:
                if verbose:
                    print(f"Cost after {iteration} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    steps.append(iteration)
        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
