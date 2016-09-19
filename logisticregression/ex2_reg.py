import numpy
from plotdata import plotdata
import matplotlib.pyplot as plt
from mapfeature import mapfeature
from costfunctionreg import costfunction, gradient


data = numpy.loadtxt("ex2data2.txt", delimiter=',')
X = data[:, :2]
y = data[:, 2:]

plotdata(X, y)
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend(["y = 1", "y = 0"])
plt.show()

m, n = X.shape
X = mapfeature(X[:, :1], X[:, 1:])
initial_theta = numpy.zeros(n+1)
lamb = 1
cost = costfunction(initial_theta, X, y, lamb)
grad = gradient(initial_theta, X, y, lamb)
print("Cost at initial theta (zeros): {:.f}".format(cost))
