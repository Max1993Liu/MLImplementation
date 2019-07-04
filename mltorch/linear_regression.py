import numpy as np

from .regression_base import RegressionBase
from .metrics import mean_square_error


class LinearRegression(RegressionBase):
    
    def _loss(self, X, y):
        prediction = X @ self.weight
        loss = mean_square_error(y, prediction)
        return self.add_penalty(loss, self.weight)

    def _predict(self, X):
        return np.dot(self.add_intercept(X), self.weight)
