import numpy


def mapfeature(X1, X2):
    X1 = numpy.matrix(X1)
    X2 = numpy.matrix(X2)
    degree = 6
    out = numpy.ones((len(X1), 1))

    for i in range(degree):
        for j in range(i):
            out = numpy.c_[out, numpy.multiply(numpy.power(X1, i-j), numpy.power(X2, j))]

    return out
