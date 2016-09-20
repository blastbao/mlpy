import numpy
from sigmoid import sigmoid


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    Theta1 = numpy.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],
                           (hidden_layer_size, input_layer_size+1), order="F")
    Theta2 = numpy.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],
                           (num_labels, hidden_layer_size+1), order="F")
    m = len(X)
    yvec = numpy.zeros((m, num_labels))
    X = numpy.c_[numpy.ones((m, 1)), X]
    a2 = sigmoid(numpy.dot(X, Theta1.T))
    a2 = numpy.c_[numpy.ones((len(a2), 1)), a2]
    a3 = sigmoid(numpy.dot(a2, Theta2.T))
    first = numpy.multiply(yvec, numpy.log(a3))
    second = numpy.multiply(1 - yvec, 1 - a3)
    cost = -numpy.sum(numpy.sum(first + second)) / m
    theta1_reg = numpy.sum(numpy.sum(numpy.power(Theta1[:, 1:], 2)))
    theta2_reg = numpy.sum(numpy.sum(numpy.power(Theta2[:, 1:], 2)))
    extra = lamb * (theta1_reg + theta2_reg) / (2 * m)

    return cost + extra


def nn_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    Theta1 = numpy.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],
                           (hidden_layer_size, input_layer_size+1), order="F")
    Theta2 = numpy.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],
                           (num_labels, hidden_layer_size+1), order="F")
    m = len(X)
    yvec = numpy.zeros((m, num_labels))
    X = numpy.c_[numpy.ones((m, 1)), X]
    a2 = sigmoid(numpy.dot(X, Theta1.T))
    a2 = numpy.c_[numpy.ones((len(a2), 1)), a2]
    a3 = sigmoid(numpy.dot(a2, Theta2.T))
    delta3 = a3 - y
    delta2 = numpy.multiply(numpy.dot(delta3, Theta2), numpy.multiply(a2, 1-a2))
    theta1_grad = numpy.dot(X.T, delta2)
    theta2_grad = numpy.dot(a2.T, delta3)

    return numpy.concatenate((theta1_grad.reshape(theta1_grad.size, order="F"),
                              theta2_grad.reshape(theta2_grad.size, order="F")))