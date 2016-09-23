import numpy
import numpy.linalg
from nncostfunction import nn_cost_function, nn_gradient
from debuginitializeweights import debug_initialize_weights
from computenumericalgradient import compute_numerical_gradient


def check_nn_gradients(lamb):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + numpy.mod(range(m), num_labels).T
    nn_params = numpy.concatenate((theta1.reshape(theta1.size, order="F"), theta2.reshape(theta2.size, order="F")))
    grad = nn_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)

    def costFunc(p):
        return nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)

    numgrad = compute_numerical_gradient(costFunc, nn_params)
    diff = numpy.linalg.norm(numgrad-grad) / numpy.linalg.norm(numgrad+grad)
    print("Relatvie difference (should be less than 1e-9):", diff)
