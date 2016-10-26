import csv
import numpy as np
from tensorflow.contrib.learn import DNNClassifier, infer_real_valued_columns_from_input


def formalize(y, num_labels):
    m = len(y)
    yvec = np.zeros((m, num_labels))
    for i in range(m):
        yvec[i, int(y.item(i))-1] = 1

    return yvec.astype(np.int32)


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
    return to_int(data), formalize(to_int(labels), 10)


def load_test_data():
    l = []
    with open("test.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    return to_int(l)


train_images, train_labels = load_train_data()
test_images = load_test_data()
print(train_images[0])

feature_columns = infer_real_valued_columns_from_input(train_images)
clf = DNNClassifier([100], feature_columns, n_classes=10)
print(train_images.shape)
print(train_labels.shape)
clf.fit(train_images, train_labels)
print("done training")

pred = clf.predict(test_images[0])
print(pred)
