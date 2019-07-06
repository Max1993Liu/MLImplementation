import numpy as np

from .activation import softmax


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier P(Yi|X1, X2) = P(Yi|X1) * P(Yi|X2)
    with P(Yi|X1) = P(Yi, X1) / P(X1) and P(Yi, X1) = P(X1|Yi) * P(Yi)
    """
    def __init__(self, categorical=False, laplace=1):
        self.categorical = categorical
        self.laplace = laplace

    def fit(self, X, y):
        assert set(y) == set([0, 1]), 'Only supports for label size 2.'
        self.n_feature = X.shape[1]

        if self.categorical:
            self._fit_categorical(X, y)
        else:
            self._fit_continuous(X, y)

    def _fit_categorical(self, X, y)
        # cnt_x is a nested list, where cnt_x[i][j] is a dictionary mapping value in jth feature to its count
        self.cnt_x = [[]] * 2
        self.p_y = np.empty(2)

        for label in range(2):
            x_label = X[y==label]

            self.p_y[label] = x_label.shape[0] / X.shape[0]

            for fet_idx in range(self.n_feature):
                fet = x_label[:, fet_idx]
                value, cnt = np.unique(fet, return_counts=True)
                cnt = cnt + self.laplace
                self.cnt_x[label].append(dict(zip(value, cnt))) 

    def _fit_continuous(self, X, y):
        self.mean_ = np.empty((2, self.n_feature))
        self.var_ = np.empty((2, self.n_feature))
        self.p_y = np.empty(2)

        n_sample = X.shape[0]

        for label in range(2):
            x_label = X[y==label]

            self.mean_[label, :] = np.mean(X, axis=0)
            self.var_[label, :] = np.var(X, axis=0)
            self.p_y[label] = x_label.shape[0] / n_sample

    def predict(self, X):
        if self.categorical:
            self._predict_categorical(X)
        else:
            return self._predict_continuous(X)

    def _predict_continuous(self, X):
        prediction = np.apply_along_axis(self._predict_single_x_continuous, 1, X)
        return softmax(prediction)

    def _predict_single_x_continuous(self, x):
        output = []
        for label in range(2):
            p_y = self.p_y[label]
            
            mean = self.mean_[label]
            var = self.var_[label]
            p_x = np.product(self._pdf(mean, var, x))

            prediction = p_y * p_x
            output.append(prediction)
        return output

    def _pdf(self, mean, var, x):
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_categorical(self, X):
        prediction = np.apply_along_axis(self._predict_single_x_categorical, 1, X)
        return softmax(prediction)

    def _predict_single_x_categorical(self, x):
        output = []

        for label in range(2):
            probabilities = []

            for fet_idx in range(self.n_feature):
                proba = self.cnt_x[label][fet_idx].get(x[fet_idx], self.laplace) / sum(self.cnt_x[label][fet_idx].values())
                probabilities.append(proba)

            output = np.product(probabilities)
        return output
