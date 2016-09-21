import numpy


def sigmoid_gradient(z):
    sig = 1 / (1 + numpy.exp(-z))

    return numpy.multiply(sig, 1 - sig)
