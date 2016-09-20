from __future__ import division
import numpy


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))
