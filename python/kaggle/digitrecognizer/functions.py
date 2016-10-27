import numpy as np
import tensorflow as tf


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def next_batch(x, y, batch_size):
    assert len(x) == len(y)
    iarr = np.arange(len(x))
    np.random.shuffle(iarr)
    iarr = iarr[:batch_size]
    return x[iarr], y[iarr]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
