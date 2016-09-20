import numpy


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))
