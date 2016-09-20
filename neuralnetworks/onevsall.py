from lrcostfunction import lr_cost_function, lr_gradient
import numpy
import scipy.optimize as op


def one_vs_all(X, y, num_labels, lamb):
    m, n = X.shape
    X = numpy.c_[numpy.ones((m, 1)), X]
    all_theta = numpy.zeros((num_labels, n+1))

    for i in range(num_labels):
        all_theta[i, :], _, _ = op.fmin_tnc(func=lr_cost_function, x0=all_theta[i, :], args=(X, (y == i+1), lamb), fprime=lr_gradient)

    return all_theta
