import numpy as np
import sigmoid as s
import sigmoidgradient as sg


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                        (hidden_layer_size, input_layer_size + 1), order='F')

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                        (num_labels, hidden_layer_size + 1), order='F')

    m = len(X)
    X = np.column_stack((np.ones((m, 1)), X))  # = a1
    a2 = s.sigmoid(np.dot(X, Theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))
    a3 = s.sigmoid(np.dot(a2, Theta2.T))
    labels = y
    y = np.zeros((m, num_labels))
    for i in range(m):
        y[i, labels[i] - 1] = 1

    cost = 0
    for i in range(m):
        cost += np.sum(y[i] * np.log(a3[i]) + (1 - y[i]) * np.log(1 - a3[i]))

    J = -(1.0 / m) * cost

    sumOfTheta1 = np.sum(np.sum(Theta1[:, 1:] ** 2))
    sumOfTheta2 = np.sum(np.sum(Theta2[:, 1:] ** 2))

    J = J + ((lambda_reg / (2.0 * m)) * (sumOfTheta1 + sumOfTheta2))

    bigDelta1 = 0
    bigDelta2 = 0

    for t in range(m):

        ## step 1: perform forward pass
        # set lowercase x to the t-th row of X
        x = X[t]
        # note that uppercase X already included column of ones
        # as bias unit from input layer to second layer, so no need to add it

        # calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
        a2 = s.sigmoid(np.dot(x, Theta1.T))

        # add column of ones as bias unit from second layer to third layer
        a2 = np.concatenate((np.array([1]), a2))
        # calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
        a3 = s.sigmoid(np.dot(a2, Theta2.T))

        ## step 2: for each output unit k in layer 3, set delta_{k}^{(3)}
        delta3 = np.zeros((num_labels))

        # see handout for more details, but y_k indicates whether
        # the current training example belongs to class k (y_k = 1),
        # or if it belongs to a different class (y_k = 1)
        for k in range(num_labels):
            y_k = y[t, k]
            delta3[k] = a3[k] - y_k

        ## step 3: for the hidden layer l=2, set delta2 = Theta2' * delta3 .* sigmoidGradient(z2)
        # note that we're skipping delta2_0 (=gradients of bias units, which we don't use here)
        # by doing (Theta2(:,2:end))' instead of Theta2'
        delta2 = (np.dot(Theta2[:, 1:].T, delta3).T) * sg.sigmoid_gradient(np.dot(x, Theta1.T))

        ## step 4: accumulate gradient from this example
        # accumulation
        # note that
        #   delta2.shape =
        #   x.shape      =
        #   delta3.shape =
        #   a2.shape     =
        # np.dot(delta2,x) and np.dot(delta3,a2) don't do outer product
        # could do e.g. np.dot(delta2[:,None], x[None,:])
        # seems faster to do np.outer(delta2, x)
        # solution from http://stackoverflow.com/a/22950320/583834
        bigDelta1 += np.outer(delta2, x)
        bigDelta2 += np.outer(delta3, a2)

    # step 5: obtain gradient for neural net cost function by dividing the accumulated gradients by m
    Theta1_grad = bigDelta1 / m
    Theta2_grad = bigDelta2 / m

    # % REGULARIZATION FOR GRADIENT
    # only regularize for j >= 1, so skip the first column
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    Theta1_grad += (float(lambda_reg) / m) * Theta1
    Theta2_grad += (float(lambda_reg) / m) * Theta2
    Theta1_grad[:, 0] = Theta1_grad_unregularized[:, 0]
    Theta2_grad[:, 0] = Theta2_grad_unregularized[:, 0]

    # Unroll gradients
    grad = np.concatenate(
        (Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))

    return J, grad
