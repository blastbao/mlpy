import numpy
import matplotlib.pyplot as plt
from costfunction import costfunction, gradient
import scipy.optimize as op
from sigmoid import sigmoid
from plotdecisionboundary import plotdecisionboundary
from predict import predict
from plotdata import plotdata

data = numpy.loadtxt("ex2data1.txt", delimiter=',')
X = data[:, :2]
y = data[:, 2:]

plotdata(X, y)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted", "Not Admitted"])
plt.show()

m, n = X.shape
X = numpy.c_[numpy.ones((m, 1)), X]
initial_theta = numpy.zeros(n+1)

cost = costfunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print("Cost:", cost.item(0))
print("Gradient at initial theta (zeros):\n", grad)

theta, _, _ = op.fmin_tnc(func=costfunction, x0=initial_theta, args=(X, y), fprime=gradient)
print("Cost:", costfunction(theta, X, y).item(0))
print("Theta:\n", theta)

nums = numpy.arange(-10, 10, 1)
plt.plot(nums, sigmoid(nums), "r")
plt.show()

plotdecisionboundary(theta, X[:, 1:], y)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted", "Not Admitted"])
plt.show()

prob = sigmoid(numpy.dot([1, 45, 85], theta.T))
print("For scores 45 and 85, the probability is", prob)

p = predict(theta, X)
print("Train accuracy: {:.1f}%".format(numpy.mean(p == y) * 100))
