import numpy
import scipy.io
from nncostfunction import nn_cost_function, nn_gradient


input_layer_size = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10

mat = scipy.io.loadmat("ex4data1.mat")
X = mat["X"]
y = mat["y"]

mat = scipy.io.loadmat("ex4weights.mat")
Theta1 = mat["Theta1"]
Theta2 = mat["Theta2"]

lamb = 0
nn_params = numpy.concatenate((Theta1.reshape(Theta1.size, order="F"), Theta2.reshape(Theta2.size, order="F")))
cost = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
print("Cost at parameters (loaded from ex4weights.mat):", cost)
