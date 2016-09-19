import numpy
import matplotlib.pyplot as plt
from computecost import computecost
from gradientdescent import gradientdescient
from mpl_toolkits.mplot3d import Axes3D

data = numpy.loadtxt("ex1data1.txt", delimiter=',')
X = data[:, :1]
y = data[:, 1:]
plt.plot(X, y, "rx", markersize=10, label="Training Data")
plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of City in 10,000s")

m, n = X.shape
X = numpy.c_[numpy.ones((m, 1)), X]
theta = numpy.zeros((2, 1))
iterations = 1500
alpha = 0.01    # 0.0.1 0.03 01 0.3 1 ...

J = computecost(X, y, theta)
print J
theta, J_history = gradientdescient(X, y, theta, alpha, iterations)
print theta

plt.plot(X[:, [1]], numpy.dot(X, theta), '-', label="Linear Regresssion")
plt.legend(loc="upper right")
plt.show()

plt.plot(range(1, iterations+1), J_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

predict1 = numpy.dot(numpy.array([[1, 3.5]]), theta)
print predict1 * 10000
predict2 = numpy.dot(numpy.array([[1, 7]]), theta)
print predict2 * 10000

theta0_vals = numpy.linspace(-10, 10, 100)
theta1_vals = numpy.linspace(-1, 4, 100)
J_vals = numpy.zeros((100, 100))

for i in range(100):
    for j in range(100):
        t = numpy.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i][j] = computecost(X, y, t)

J_vals = J_vals.T
fig = plt.figure()
ax = fig.gca(projection="3d")
suf = ax.plot_surface(theta0_vals, theta1_vals, J_vals)
plt.xlabel("theta_0")
plt.ylabel("theta_1")
plt.show()

fig = plt.figure()
plt.contour(theta0_vals, theta1_vals, J_vals, numpy.logspace(-2, 3, 20))
plt.xlabel("theta_0")
plt.ylabel("theta_1")
plt.plot(theta.item(0), theta.item(1), "rx", markersize=10, linewidth=2)
plt.show()
