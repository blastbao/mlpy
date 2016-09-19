import numpy
from sigmoid import sigmoid


def costfunction(theta, X, y):
    m, _ = y.shape
    theta = numpy.matrix(theta)
    X = numpy.matrix(X)
    y = numpy.matrix(y)
    sig = sigmoid(numpy.dot(X, theta.T))

    return (-(numpy.dot(y.T, numpy.log(sig))) - numpy.dot((1 - y).T, numpy.log(1 - sig))) / m


def gradient(theta, X, y):
    m, n = X.shape
    theta = numpy.matrix(theta)
    X = numpy.matrix(X)
    y = numpy.matrix(y)
    error = sigmoid(X * theta.T) - y

    return numpy.sum(numpy.multiply(error, X), 0) / m

