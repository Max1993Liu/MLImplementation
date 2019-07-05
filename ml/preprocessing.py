import numpy as np


def standardize(X, axis=0):
    mean_ = np.mean(X, axis=axis, keepdims=True)
    std_ = np.std(X, axis=axis, keepdims=True)
    return (X - mean_) / (std_ + 1e-15)
