import numpy as np


def entropy(x):
    """ Calculate the entroy for a label sequence (non-negative integers) """
    cnt = np.bincount(x)
    proba = cnt / cnt.sum()
    return -np.sum([p * np.log2(p) for p in proba if p > 0])


def gini(x):
    """ Calculate the gini for a label sequence (non-negative integers) """
    cnt = np.bincount(x)
    proba = cnt / cnt.sum()
    return 1 - np.sum([p ** 2 for p in proba])

