import numpy as np


def mean_square_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def mean_absolute_error(y_true, y_pred):
	return (y_true, - y_pred).abs().mean()


def binary_crossentropy(y_true, y_pred, epsilon=1e-12):
    pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    cse = -(y_true * np.log(pred_clipped) + (1 - y_true) * np.log(1 - pred_clipped)).mean()
    return cse


def auc(x, y):
    """ Calculate the area under curve """
    dx = np.diff(x)
    if np.all(dx >= 0):
        multiplier = 1
    elif np.all(dx <= 0):
        multiplier = -1
    else:
        raise ValueError('x should be sorted first.')
    return multiplier * np.trapz(y, x)


def roc_curve(y_true, y_pred):
    """ Return (fpr, tpr, thresholds)"""
    assert set(y_true) == set([0, 1]), 'Only support binary labels'
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order].astype(bool)
    y_pred_sorted = y_pred[order]

    threshold_idx = np.where(np.diff(y_pred_sorted))[0]
    threshold = y_pred_sorted[threshold_idx]
    if 0 not in threshold:
        threshold = np.r_[0, threshold]
    if 1 not in threshold:
        threshold = np.r_[threshold, 1]

    tpr, fpr = [], []

    n_threshold = len(threshold)
    for idx,th in enumerate(threshold):
        # TPR = TP /（TP + FN） 
        tpr.append(((y_pred_sorted >= th) & (y_true_sorted)).sum() / y_true.sum())
        # FPR = FP /（FP + TN） 
        fpr.append(((y_pred_sorted >= th) & (~y_true_sorted)).sum() / (~y_true_sorted).sum())    

    return tpr, fpr, threshold


def roc_auc_score(y_true, y_pred):
	tpr, fpr, _ = roc_curve(y_true, y_pred)
	return auc(fpr, tpr)
