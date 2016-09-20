import numpy
import matplotlib.pyplot as plt


def plotdata(X, y):
    pos = numpy.where(y == 1)[0]
    neg = numpy.where(y == 0)[0]

    plt.plot(X[pos, 0], X[pos, 1], "k+", linewidth=2, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], "ko", markerfacecolor="y", markersize=7)
