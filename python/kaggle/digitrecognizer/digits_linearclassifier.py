from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import csv
import numpy as np
import tensorflow as tf

ARGFLAGS = None
DATA_SETS = None
SPLIT_INDEX = 35000

# comment out for less info during the training runs.
tf.logging.set_verbosity(tf.logging.INFO)


def define_and_run_dnn_classifier(images, labels):
    """Run a DNN classifier."""
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(images)
    classifier = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns, n_classes=10,
        # After you've done a training run with optimizer learning rate 0.1,
        # change it to 0.5 and run the training again.  Use TensorBoard to take
        # a look at the difference.  You can see both runs by pointing it to
        # the parent model directory, which by default is:
        #   tensorboard --logdir=/tmp/tfmodels/mnist_tflearn
        optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.01,
            l1_regularization_strength=0.001
        ),
        model_dir=ARGFLAGS.model_dir,
        enable_centered_bias=False
        )
    classifier.fit(images, labels.astype(np.int64), batch_size=1000, max_steps=ARGFLAGS.num_steps)
    print("Finished running the training via the fit() method")
    return classifier


def eval_dnn_classifier(classifier, images, labels):
    # Evaluate classifier accuracy.
    accuracy_score = classifier.evaluate(images, labels.astype(np.int64))['accuracy']
    print('LinearClassifier Accuracy: {0:f}'.format(accuracy_score))


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
    return to_int(data), to_int(labels)


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
    print("\n---- Running DNN classifier...")
    classifier = define_and_run_dnn_classifier(train_images, train_labels)
    print("\n---Evaluating DNN classifier accuracy...")
    eval_dnn_classifier(classifier, validation_images, validation_labels)
    test_images = load_test_data()
    result = classifier.predict(test_images)
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
