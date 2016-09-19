import numpy
from featurenormalize import featurenormalize
from gradientdescentmulti import gradientdescientmulti
import matplotlib.pyplot as plt


data = numpy.loadtxt("ex1data2.txt", delimiter=',')
X = data[:, :2]
y = data[:, 2:]
m, _ = X.shape

print "First 10 exmpales from the dataset:"
print data[:10, :]

X, mu, sigma = featurenormalize(X)
X = numpy.c_[numpy.ones((m, 1)), X]

alpha = 0.01
iterations = 1500

theta = numpy.zeros((3, 1))
theta, J_history = gradientdescientmulti(X, y, theta, alpha, iterations)

plt.figure()
plt.plot(range(1, iterations+1), J_history, "-r", linewidth=2)
plt.xlabel("Number of iterations")
plt.ylabel("Cost J")
plt.show()

print "Theta computed from gradient descent:\n", theta

price = numpy.dot(numpy.c_[[1], (numpy.array([[1650, 3]]) - mu) / sigma], theta)
print "Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):", price

from sklearn.linear_model import LinearRegression

regression = LinearRegression(normalize=True)
regression.fit(data[:, :2], data[:, 2:])
pred = regression.predict([[1650, 3]])
print "Predicted price by sklearn (LinearRegression):", pred


# normal equation
X = numpy.c_[numpy.ones((m, 1)), data[:, :2]]
theta = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X.T, X)), X.T), y)
price = numpy.dot(numpy.array([[1, 1650, 3]]), theta)
print "Predicted price by normal equation:", price
