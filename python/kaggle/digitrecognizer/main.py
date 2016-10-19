import csv
import numpy as np
import neuralnetwork as network


def to_int(arr):
    mat = np.mat(arr)
    m, n = mat.shape
    ret = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ret[i, j] = int(mat[i, j])
    return ret


def load_train_data():
    l = []
    with open("train.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = np.array(l)
    labels = l[:, 0]
    data = l[:, 1:]
    return to_int(data), to_int(labels).T


def load_test_data():
    l = []
    with open("test.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    return to_int(np.array(l))


train_images, train_labels = load_train_data()
train_images, validation_images = train_images[:, :3000], train_images[:, 3000:]
train_labels, validation_labels = train_labels[:, :3000], train_labels[:, 3000:]
test_images = load_test_data()

net = network.Network([784, 100, 10])
net.stochastic_gradient_descent(train_images, 30, 10, 0.5,
                                evaluation_data=validation_images,
                                monitor_evaluation_accuracy=True)
print net.predict(test_images)
