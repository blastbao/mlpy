from sigmoid import sigmoid
import numpy


def predict(theta, X):
    theta = numpy.matrix(theta)
    X = numpy.matrix(X)
    sig = sigmoid(X.dot(theta.T))

    return sig >= 0.5
