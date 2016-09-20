import numpy


def featurenormalize(X):
    mu = X.mean(axis=0)
    sigma = numpy.std(X, ddof=1)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma