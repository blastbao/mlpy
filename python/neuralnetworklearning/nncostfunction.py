import numpy
from sigmoid import sigmoid


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    Theta1 = numpy.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],
                           (hidden_layer_size, input_layer_size+1), order="F")
    Theta2 = numpy.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],
                           (num_labels, hidden_layer_size+1), order="F")
    m = len(X)
    yvec = __formalize(y, num_labels)
    X = numpy.c_[numpy.ones((m, 1)), X]
    a2 = sigmoid(numpy.dot(X, Theta1.T))
    a2 = numpy.c_[numpy.ones((len(a2), 1)), a2]
    a3 = sigmoid(numpy.dot(a2, Theta2.T))
    first = numpy.multiply(yvec, numpy.log(a3))
    second = numpy.multiply(1 - yvec, numpy.log(1 - a3))
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
    yvec = __formalize(y, num_labels)
    X = numpy.c_[numpy.ones((m, 1)), X]
    a2 = sigmoid(numpy.dot(X, Theta1.T))
    a2 = numpy.c_[numpy.ones((len(a2), 1)), a2]
    a3 = sigmoid(numpy.dot(a2, Theta2.T))
    delta3 = a3 - yvec
    delta2 = numpy.multiply(numpy.dot(delta3, Theta2[:, 1:]), numpy.multiply(a2[:, 1:], 1-a2[:, 1:]))
    theta1_grad = numpy.dot(delta2, X)
    theta2_grad = numpy.dot(delta3.T, a2)

    print(numpy.sum(numpy.sum(theta1_grad)))
    print(numpy.sum(numpy.sum(theta2_grad)))

    reg1 = lamb * Theta1[:, 1:] / m
    reg2 = lamb * Theta2[:, 1:] / m
    theta1_grad += numpy.c_[numpy.zeros((len(reg1), 1)), reg1]
    theta2_grad += numpy.c_[numpy.zeros((len(reg2), 1)), reg2]

    return numpy.concatenate((theta1_grad.reshape(theta1_grad.size, order="F"),
                              theta2_grad.reshape(theta2_grad.size, order="F")))


def __formalize(y, num_labels):
    m = len(y)
    yvec = numpy.zeros((m, num_labels))
    for i in range(m):
        yvec[i, y.item(i)-1] = 1

    return yvec
