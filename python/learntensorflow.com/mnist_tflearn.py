from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

ARGFLAGS = None
DATA_SETS = None

# comment out for less info during the training runs.
tf.logging.set_verbosity(tf.logging.INFO)


def run_linear_classifier():
    """Run a linear classifier."""
    global DATA_SETS
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        DATA_SETS.train.images)
    classifier = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns, n_classes=10)
    classifier.fit(DATA_SETS.train.images,
                   DATA_SETS.train.labels.astype(numpy.int64),
                   batch_size=100, steps=10000)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(
        DATA_SETS.test.images,
        DATA_SETS.test.labels.astype(numpy.int64))['accuracy']
    print('Linear Classifier Accuracy: {0:f}'.format(accuracy_score))


def define_and_run_dnn_classifier():
    """Run a DNN classifier."""
    global DATA_SETS
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        DATA_SETS.train.images)
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns, n_classes=10,
        hidden_units=[128, 32],
        # After you've done a training run with optimizer learning rate 0.1,
        # change it to 0.5 and run the training again.  Use TensorBoard to take
        # a look at the difference.  You can see both runs by pointing it to
        # the parent model directory, which by default is:
        #   tensorboard --logdir=/tmp/tfmodels/mnist_tflearn
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1),
        model_dir=ARGFLAGS.model_dir
        )
    classifier.fit(DATA_SETS.train.images,
                   DATA_SETS.train.labels.astype(numpy.int64),
                   batch_size=100, max_steps=ARGFLAGS.num_steps)
    print("Finished running the training via the fit() method")
    return classifier


def eval_dnn_classifier(classifier):
    # Evaluate classifier accuracy.
    global DATA_SETS
    accuracy_score = classifier.evaluate(
        DATA_SETS.test.images,
        DATA_SETS.test.labels.astype(numpy.int64))['accuracy']
    print('DNN Classifier Accuracy: {0:f}'.format(accuracy_score))


def main(_):

    # read in data, downloading first if necessary
    global DATA_SETS
    print("Downloading and reading data sets...")
    DATA_SETS = input_data.read_data_sets(ARGFLAGS.data_dir)
    print("train images:", DATA_SETS.train.images.shape)
    print("train labels:", DATA_SETS.train.labels.shape)
    print("test images:", DATA_SETS.test.images.shape)
    print("test labels:", DATA_SETS.test.labels.shape)

    # print("\n-----Running linear classifier...")
    # run_linear_classifier()

    print("\n---- Running DNN classifier...")
    classifier = define_and_run_dnn_classifier()
    print("\n---Evaluating DNN classifier accuracy...")
    eval_dnn_classifier(classifier)


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