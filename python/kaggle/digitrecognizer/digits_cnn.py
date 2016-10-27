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
BATCH_SIZE = 1000
MAX_STEPS = 20000

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


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def main(_):
    train_images, train_labels = load_train_data()
    train_images, validation_images = train_images[:SPLIT_INDEX, :], train_images[SPLIT_INDEX:, :]
    train_labels, validation_labels = train_labels[:SPLIT_INDEX, :], train_labels[SPLIT_INDEX:, :]

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    W_conv1 = functions.weight_variable([5, 5, 1, 32])
    b_conv1 = functions.bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = functions.weight_variable([5, 5, 32, 64])
    b_conv2 = functions.bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = functions.weight_variable([7*7*64, 1024])
    b_fc1 = functions.bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = functions.weight_variable([1024, 10])
    b_fc2 = functions.bias_variable([10])
    y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    model = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(model)
        for i in range(MAX_STEPS):
            images, labels = functions.next_batch(train_images, train_labels, BATCH_SIZE)
            if i > 0 and i % 100 == 0:
                train_accuracy = session.run(accuracy, feed_dict={x: images, y: labels, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            session.run(train_step, feed_dict={x: images, y: labels, keep_prob: 0.5})
        print(session.run(accuracy, feed_dict={x: validation_images, y: validation_labels, keep_prob: 1.0}))

        test_images = load_test_data()
        result = session.run(tf.argmax(y_pred, 1), feed_dict={x: test_images, keep_prob: 1.0})
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
