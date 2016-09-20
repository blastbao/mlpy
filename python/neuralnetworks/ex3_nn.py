import numpy
import scipy.io
from predict import predict

mat = scipy.io.loadmat("ex3data1.mat")
X = mat["X"]
y = mat["y"]
m, _ = X.shape

mat = scipy.io.loadmat("ex3weights.mat")
Theta1 = mat["Theta1"]
Theta2 = mat["Theta2"]

pred = predict(Theta1, Theta2, X)
print("Training set accuracy: {:.1f}%".format(numpy.mean(pred == numpy.squeeze(y)) * 100))

rp = numpy.random.permutation(range(m))

for i in range(min(10, m)):
    pred = predict(Theta1, Theta2, X[rp[i], :]).item(0)
    print("Neural network prediction: {:d} (digit {:d})".format(pred, pred % 10))
