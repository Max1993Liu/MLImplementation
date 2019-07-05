import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def relu(x):
    return np.maximum(x, 0)


def leakyrelu(x, a=0.01):
    return np.maximum(x * a, x)
