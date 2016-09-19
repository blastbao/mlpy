import numpy


def computecost(X, y, theta):
    m, _ = X.shape
    J = 0
    diff = numpy.dot(X, theta) - y
    arr = numpy.dot(diff.T, diff) / (2 * m)
    return arr.item(0)
