import numpy
from costfunction import costfunction, gradient


def costfunctionreg(theta, X, y, lamb):
    m, n = X.shape
    cost = costfunction(theta, X, y)
    reg = lamb * numpy.sum(numpy.power(theta[1:], 2)) / (2 * m)

    return cost + reg


def gradientreg(theta, X, y, lamb):
    m, n = X.shape
    grad = gradient(theta, X, y)
    theta = numpy.matrix(theta)
    reg = lamb * numpy.r_[numpy.zeros((1, len(theta))), theta.T[1:, :]] / m

    return grad + reg.flatten()
