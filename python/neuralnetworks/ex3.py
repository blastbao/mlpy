import numpy
import scipy.io
from onevsall import one_vs_all
from predictonevsall import predict_one_vs_all
from displayData import displayData

mat = scipy.io.loadmat("ex3data1.mat")
X = mat["X"]
y = mat["y"]
m, _ = X.shape

# rand_indices = numpy.random.permutation(range(m))
# sel = X[rand_indices[0:100], :]

# displayData(sel)

lamb = 0.1
num_labels = 10

all_theta = one_vs_all(X, y, num_labels, lamb)

pred = predict_one_vs_all(all_theta, X)
print("Training set accuracy: {:.1f}%".format(numpy.mean(pred == numpy.squeeze(y)) * 100))
