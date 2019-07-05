import numpy as np


def mean_square_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def mean_absolute_error(y_true, y_pred):
	return (y_true, - y_pred).abs().mean()


def binary_crossentropy(y_true, y_pred, epsilon=1e-12):
    pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    cse = -(y_true * np.log(pred_clipped) + (1 - y_true) * np.log(1 - pred_clipped)).mean()
    return cse

