from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import csv
import numpy as np
import tensorflow as tf
import functions
from tensorflow.python.ops import rnn, rnn_cell


ARGFLAGS = None
DATA_SETS = None
SPLIT_INDEX = 35000
BATCH_SIZE = 100
MAX_STEPS = 2000

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# comment out for less info during the training runs.
tf.logging.set_verbosity(tf.logging.INFO)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def to_int(arr):
    mat = np.mat(arr)
    m, n = mat.shape
    ret = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ret[i, j] = mat[i, j]
    return ret.astype(np.int32)


def load_train_data():
    l = []
    with open("train.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = np.array(l)
    np.random.shuffle(l)
    labels = l[:, :1]
    data = l[:, 1:]
    return to_int(data), functions.dense_to_one_hot(to_int(labels), 10)


def load_test_data():
    l = []
    with open("test.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    return to_int(l)


def save_result(result):
    with open("result.csv", "w") as f:
        f.write("ImageId,Label\n")
        for idx, value in enumerate(result):
            f.write("{},{}\n".format(idx+1, value))


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def main(_):
    pred = RNN(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    train_images, train_labels = load_train_data()
    train_images, validation_images = train_images[:SPLIT_INDEX, :], train_images[SPLIT_INDEX:, :]
    train_labels, validation_labels = train_labels[:SPLIT_INDEX, :], train_labels[SPLIT_INDEX:, :]

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = functions.next_batch(train_images, train_labels, batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        test_images = load_test_data()
        timages = test_images.reshape((test_images.shape[0], n_steps, n_input))
        result = sess.run(tf.argmax(pred, 1), feed_dict={x: timages})
        save_result(result)
        print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='MNIST_data',
                        help='Directory for storing data')
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(
                            "/tmp/tfmodels/mnist_tflearn",
                            str(int(time.time()))),
                        help='Directory for storing model info')
    parser.add_argument('--num_steps', type=int,
                        default=25000,
                        help='Number of training steps to run')
    ARGFLAGS = parser.parse_args()
    tf.app.run()
