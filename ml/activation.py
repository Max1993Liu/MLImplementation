import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(z):
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def relu(x):
    return np.maximum(x, 0)


def leakyrelu(x, a=0.01):
    return np.maximum(x * a, x)
