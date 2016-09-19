import numpy
from plotdata import plotdata
import matplotlib.pyplot as plt
from mapfeature import mapfeature
from costfunctionreg import costfunctionreg, gradientreg
import scipy.optimize as op
from plotdecisionboundary import plotdecisionboundary


data = numpy.loadtxt("ex2data2.txt", delimiter=',')
X = data[:, :2]
y = data[:, 2:]

plotdata(X, y)
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend(["y = 1", "y = 0"])
plt.show()

X = mapfeature(X[:, :1], X[:, 1:])
m, n = X.shape
initial_theta = numpy.zeros(n)
lamb = 1
cost = costfunctionreg(initial_theta, X, y, lamb)
grad = gradientreg(initial_theta, X, y, lamb)
print("Cost at initial theta (zeros): ", cost.item(0))

initial_theta = numpy.zeros(n)
lamb = 1
theta, _, _ = op.fmin_tnc(func=costfunctionreg, x0=initial_theta, args=(X, y, lamb), fprime=gradientreg)
plotdecisionboundary(theta, X[:, 1:], y)
plt.title("lambda = {:.2f}".format(lamb))
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend(["y = 1", "y = 0", "Decision boundary"])
plt.show()
