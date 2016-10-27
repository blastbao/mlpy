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


ARGFLAGS = None
DATA_SETS = None
SPLIT_INDEX = 35000

# comment out for less info during the training runs.
tf.logging.set_verbosity(tf.logging.INFO)


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


def main(_):
    train_images, train_labels = load_train_data()
    train_images, validation_images = train_images[:SPLIT_INDEX, :], train_images[SPLIT_INDEX:, :]
    train_labels, validation_labels = train_labels[:SPLIT_INDEX, :], train_labels[SPLIT_INDEX:, :]

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_pred = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    session = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    for _ in range(25000):
        images, labels = functions.next_batch(train_images, train_labels, 100)
        session.run(train_step, feed_dict={x: images, y: labels})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accuracy, feed_dict={x: validation_images, y: validation_labels}))

    test_images = load_test_data()
    result = session.run(tf.argmax(y_pred, 1), feed_dict={x: test_images})
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
