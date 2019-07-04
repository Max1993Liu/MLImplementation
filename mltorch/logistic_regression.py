import numpy as np

from .regression_base import RegressionBase
from .metrics import binary_crossentropy
from .activation import sigmoid


class LogisticRegression(RegressionBase):

    def __init__(self, C=0.01, penalty='l1', lr=1e-4, max_iters=1000, tolerance=1e-15):
        self.C = C
        self.penalty = penalty
        self.lr = lr
        self.max_iters = max_iters
        self.weight = None
        self.tolerance = tolerance

    def _loss(self, X, y):
        prediction = X @ self.weight
        proba = sigmoid(prediction)
        loss = binary_crossentropy(y, proba)
        return self.add_penalty(loss, self.weight)

    def _predict(self, X):
        prediction = np.dot(self.add_intercept(X), self.weight)
        return 1 / (1 + np.exp(-prediction))

