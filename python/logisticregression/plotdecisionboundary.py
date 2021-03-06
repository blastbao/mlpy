import matplotlib.pyplot as plt
import numpy
from mapfeature import mapfeature
from plotdata import plotdata


def plotdecisionboundary(theta, X, y):
    plotdata(X[:, :2], y)

    if X.shape[1] <= 2:
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min = -(theta[0, 0] + theta[0, 1]*x_min) / theta[0, 2]
        y_max = -(theta[0, 0] + theta[0, 1]*x_max) / theta[0, 2]
        plt.plot([x_min, x_max], [y_min, y_max])
        plt.axis([30, 100, 30, 100])
    else:
        u = numpy.linspace(-1, 1.5, 50)
        v = numpy.linspace(-1, 1.5, 50)
        z = numpy.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = numpy.dot(mapfeature(u[i].T, v[j].T), theta.T)

        z = z.T
        plt.contour(u, v, z, [0.5], linewidth=2).collections[0]
