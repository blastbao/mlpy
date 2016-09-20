import numpy
from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    Theta1 = numpy.matrix(Theta1)
    Theta2 = numpy.matrix(Theta2)
    X = numpy.matrix(X)
    m = len(X)

    X = numpy.c_[numpy.ones((m, 1)), X]
    A1 = sigmoid(numpy.dot(X, Theta1.T))
    A1 = numpy.c_[numpy.ones((len(A1), 1)), A1]
    A2 = sigmoid(numpy.dot(A1, Theta2.T))

    return numpy.argmax(A2, axis=1) + 1