import numpy as np
import scipy

from .preprocessing import standardize



class PCA:

    def __init__(self, n_component, random_state=None):
        self.n_component = n_component
        self.components = None
        self.variance_ratio = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y=None):
        self._solve_eigen(X)

    def _solve_eigen(self, X):
        cov_matrix = X.T @ X
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)
        
        variance = np.square(eig_values) / (X.shape[0] - 1)
        variance_ratio = variance / variance.sum()

        self.components = eig_vectors[:, :self.n_component]
        self.variance_ratio = variance_ratio[:self.n_component]

    def transform(self, X):
        assert self.components is not None, 'Call the fit method first.'
        return X @ self.components
