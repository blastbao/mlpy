import argparse
from keras.models import Model
from keras.layers import Dense, Input
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-i", nargs="?")
parser.add_argument("-t", nargs="?")
parser.add_argument("-n", type=int, default=100)
args = parser.parse_args()

df = pd.read_csv(args.i, header=None)
x = df.ix[:, 1:]
y = df.ix[:, :0]
l1 = Input((1,))
l2 = Dense(1, activation="linear")(l1)
model = Model(l1, l2)
model.compile(loss="mse", optimizer="sgd")
model.fit(x.values, y.values, nb_epoch=args.n, validation_split=0.2)

test_x = pd.read_csv(args.t, header=None)
result = model.predict(test_x.values)
print(result)
