from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse


FLAGS = None


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[100],
                                                n_classes=10,
                                                model_dir="/tmp/mnist_model")

    # Fit model.
    classifier.fit(x=mnist.train.images, y=mnist.train.labels, steps=2000)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=mnist.test.images, y=mnist.test.labels)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data', help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()
