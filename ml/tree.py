import numpy as np


def entropy(x):
    """ Calculate the entroy for a label sequence (non-negative integers) """
    cnt = np.bincount(x)
    proba = cnt / cnt.sum()
    return -np.sum([p * np.log2(p) for p in proba if p > 0])


def information_gain(x, x_split):
    """ x_split is a list of label sequence """
    split_entropy = sum([entropy(split) * split.shape[0] / x.shape[0] for split in x_split])
    return entropy(x) - split_entropy


def gini(x):
    """ Calculate the gini for a label sequence (non-negative integers) """
    cnt = np.bincount(x)
    proba = cnt / cnt.sum()
    return 1 - np.sum([p ** 2 for p in proba])


class DecisionTreeBase:

    def __init__(self, criterion=None):
        self.criterion = criterion
        # gini or entropy at this node
        self.node_stats = None
        self.threshold = None
        self.column_index = None
        # only available at leaf node
        self.outcome = None

        self.left_child = None
        self.right_child = None


    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    @staticmethod
    def split(X, y, column_index, threshold):
        left_mask = X[:, column_index] < threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _find_splits(self, x):
        """ Find possible split values in a 1-d array x in ascending order """
        split_values = set()

        x = sorted(set(x))
        for i in range(len(x) - 1):
            split_values.add((x[i] + x[i+1]) / 2)

        return list(split_values)

    def _find_best_split(self, X, y):
        """ Find the best split across all features, 
            return the column_index and threshold for the best split """
        best_stats, best_col, best_thresh = None, None, None

        for column_index in range(X.shape[1]):
            
            candidate_splits = self._find_splits(X[:, column_index])
            for threshold in candidate_splits:
                _, _, y_left, y_right = self.split(X, y, column_index, threshold)
                stats = self.criterion(y, [y_left, y_right])


