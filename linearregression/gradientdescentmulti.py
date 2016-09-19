import numpy
from computecostmulti import computecostmulti


def gradientdescientmulti(X, y, theta, alpha, iterations):
    m, _ = X.shape
    J_history = numpy.zeros((iterations, 1))

    for it in range(iterations):
        diff = numpy.dot(X, theta) - y
        theta = theta - (alpha / m * numpy.dot(diff.T, X)).T
        J_history[it] = computecostmulti(X, y, theta)

    return theta, J_history