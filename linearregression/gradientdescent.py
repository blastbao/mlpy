import numpy
from computecost import computecost


def gradientdescient(X, y, theta, alpha, iterations):
    m, _ = X.shape
    J_history = numpy.zeros((iterations, 1))

    for it in range(iterations):
        diff = numpy.dot(X, theta) - y
        theta = theta - (alpha / m * numpy.dot(diff.T, X)).T
        J_history[it] = computecost(X, y, theta)

    return theta, J_history