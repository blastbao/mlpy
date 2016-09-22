import numpy


def debug_initialize_weights(fanout, fanin):
    z = numpy.zeros((fanout, fanin + 1))

    return numpy.reshape(numpy.sin(range(z.size)), z.shape) / 10
