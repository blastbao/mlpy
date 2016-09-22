import numpy
import scipy.io
from nncostfunction import nn_cost_function, nn_gradient
from sigmoidgradient import sigmoid_gradient
from randinitializeweights import rand_initialize_weights
from checknngradients import check_nn_gradients


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

lamb = 1
cost = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
print("Cost at parameters (loaded from ex4weights.mat):", cost)

g = sigmoid_gradient(numpy.array([1, -0.5, 0, 0.5, 1]))
print(g)

initial_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
initial_nn_params = numpy.concatenate((initial_theta1.reshape(initial_theta1.size, order="F"),
                                       initial_theta2.reshape(initial_theta2.size, order="F")))
print(initial_nn_params)

check_nn_gradients(0)
