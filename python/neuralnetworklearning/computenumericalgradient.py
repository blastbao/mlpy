import numpy


def compute_numerical_gradient(cost, theta):
    numgrad = numpy.zeros(theta.shape)
    perturb = numpy.zeros(theta.shape)
    e = 1e-4

    for p in range(theta.size):
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = cost(theta - perturb)
        loss2, _ = cost(theta + perturb)
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad
