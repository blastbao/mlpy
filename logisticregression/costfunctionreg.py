import numpy
from costfunction import costfunction as costfn, gradient as gradfn


def costfunction(theta, X, y, lamb):
    cost = costfn(theta, X, y)
    reg = lamb * numpy.sum(numpy.power(theta[1:], 2)) / (2 * m)

    return cost + reg


def gradient(theta, X, y, lamb):
    m, n = X.shape
    grad = gradfn(theta, X, y)
    reg = lamb * numpy.r_[numpy.zeros((1, n)), theta[1:, :]] / m

    return grad + reg
