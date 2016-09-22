import numpy.random


def rand_initialize_weights(lin, lout):
    init_epsilon = 0.12  # sqrt(6) / sqrt(lin + lout)

    return numpy.random.rand(lout, lin + 1) * 2 * init_epsilon - init_epsilon
