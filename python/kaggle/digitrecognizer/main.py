import csv
import numpy as np
import neuralnetwork as network


def formalize(y, num_labels):
    m = len(y)
    print(m)
    yvec = np.zeros((m, num_labels))
    for i in range(m):
        yvec[i, int(y.item(i))-1] = 1

    return yvec


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
    with open("t.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = np.matrix(l)
    labels = l[:, 0]
    data = l[:, 1:]
    return to_int(data), formalize(to_int(labels), 10)


def load_test_data():
    l = []
    with open("tt.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    return to_int(np.matrix(l))


train_images, train_labels = load_train_data()
print(train_images)
print(train_labels)
# train_images, validation_images = train_images[:, :3000], train_images[:, 3000:]
train_images, validation_images = train_images[0, :], train_images[1, :]
# train_labels, validation_labels = train_labels[:, :3000], train_labels[:, 3000:]
train_labels, validation_labels = train_labels[0, :], train_labels[1, :]
test_images = load_test_data()

net = network.Network([784, 100, 10])
net.stochastic_gradient_descent(zip(train_images, train_labels), 30, 10, 0.5,
                                evaluation_data=zip(validation_images, validation_labels),
                                monitor_evaluation_accuracy=True)
print(net.predict(test_images))
