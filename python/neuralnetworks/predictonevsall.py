import numpy
from sigmoid import sigmoid


def predict_one_vs_all(all_theta, X):
    prob = sigmoid(numpy.dot(numpy.c_[numpy.ones((len(X), 1)), X], all_theta.T))

    return numpy.squeeze(numpy.argmax(prob, axis=1) + 1)
