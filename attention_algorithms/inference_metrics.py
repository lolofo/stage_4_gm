import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import auc


def scalar_jaccard(y_true, y_pred):
    num = np.dot(y_true, y_pred)
    den = sum(y_true) + sum(y_pred) - np.dot(y_true, y_pred)
    return num / den


def au_precision_curve(y_true, y_pred):
    tr = np.linspace(0, 1, 50)
    curve = []
    for t in tr:
        preds = 1 * (y_pred >= t)
        curve.append(precision_score(y_true, preds))

    return auc(x=tr, y=curve)


def au_recall_curve(y_true, y_pred):
    tr = np.linspace(0, 1, 50)
    curve = []
    for t in tr:
        preds = 1 * (y_pred >= t)
        curve.append(recall_score(y_true, preds))

    return auc(x=tr, y=curve)


def precision(y_true, y_pred):
    tr = 0.2
    preds = 1 * (y_pred >= tr)
    return precision_score(y_true, preds)


def recall(y_true, y_pred):
    tr = 0.2
    preds = 1 * (y_pred >= tr)
    return recall_score(y_true, preds)
