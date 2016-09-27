import json
import random
import sys
import numpy as np


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return a-y


class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta,
                                    lmbda = 0.0,
                                    evaluation_data = None,
                                    monitor_evaluation_cost = False,
                                    monitor_evaluation_accuracy = False,
                                    monitor_training_cost = False,
                                    monitor_training_accuracy = False):
        if evaluation_data:
            n_data = len(evaluation_data)

        n = len(training_data)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig * (1-sig)
