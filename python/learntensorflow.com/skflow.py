from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn import LinearClassifier


def formalize(y, num_labels):
    m = len(y)
    yvec = np.zeros((m, num_labels))
    for i in range(m):
        yvec[i, int(y.item(i))-1] = 1

    return yvec


digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
clf = LinearClassifier(feature_columns, n_classes=10)
clf.fit(X_train, y_train.astype(np.int32))
print(2)
pred = clf.predict(X_test)
print("aaaaaa")
print(np.mean(pred == y_test))
