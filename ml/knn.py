import numpy as np
from types import FunctionType, LambdaType
from collections import Counter

from .distance import euclidean_distance
from .preprocessing import standardize


class KNNBase:

    def __init__(self, k=3, distance='euclidean_distance'):
        self.k = k
        self.distance = distance
        if isinstance(distance, str):
            if distance == 'euclidean_distance':
                self.distance_fn = euclidean_distance
            else:
                raise ValueError('Only euclidean distance if supported for now.')
        elif isinstance(distance, (FunctionType, LambdaType)):
            self.distance_fn = distance
        else:
            raise ValueError('Only string or function objects are allowed as distance argument.')


    def fit(self, X, y):
        self.X = X
        self.y = y

    def _find_neighbours(self, x):
        """ Find a list of nearest neighbour labels to x """
        distance = [self.distance_fn(x, sample) for sample in self.X]
        neighbours = sorted(zip(distance, self.y))[:self.k]
        return [n[1] for n in neighbours]

    def _predict_from_neighbours(self, neighbours):
        raise NotImplementedError

    def predict(self, X):
        pred = [self._predict_from_neighbours(self._find_neighbours(x)) for x in X]
        return np.array(pred)


class KNNClassifier(KNNBase):

    def _predict_from_neighbours(self, neighbours):
        return Counter(neighbours).most_common(1)[0][0]


class KNNRegressor(KNNBase):

    def _predict_from_neighbours(self, neighbours):
        return np.mean(neighbours)

