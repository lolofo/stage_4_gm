"""
Implementation of how to plot the ROC curve
"""
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
from attention_algorithms.raw_attention import RawAttention
from attention_algorithms.attention_flow import attention_flow_max
from attention_algorithms.raw_attention import normalize_attention


# --> plot the roc_curve of the model
def plot_roc_curve(Y_test, probs):
    preds = probs
    fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    fig = plt.figure(figsize=(10, 10))
    plt.title('ROC CURVE')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

    plt.plot([0, 1], [0, 1], 'r--', label="random classifier")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right', prop={'size': 20})

    # selection of the best threshold

    best_tr = threshold[np.argmax(tpr - fpr)]

    return fig, best_tr


def combine_roc_curves(Y_test, probs,
                       legend=["max_agreg", "avg_agreg"]):
    fig = plt.figure(figsize=(10, 10))
    plt.title('ROC CURVE')
    plt.plot([0, 1], [0, 1], 'r--', label="random classifier")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    assert len(probs) == len(legend)

    for i in range(len(probs)):
        preds = probs[i]
        fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('ROC CURVE')
        plt.plot(fpr, tpr, label=f"AUC {legend[i]} : {np.round(roc_auc, 2)}")
        plt.legend(loc='lower right', prop={'size': 10})

    return fig


def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def default_plot_colormap(map,
                          xlabel, ylabel, title,
                          xstick=None,
                          sz=(10, 10)):
    # the global figure
    fig = plt.figure(figsize=sz)
    plt.imshow(map, aspect='auto', cmap='Purples')
    plt.title(title)
    ax = plt.gca()

    # the x-axis
    plt.xlabel(xlabel)
    ax.set_xticks(range(map.shape[1]))
    if xstick is None:
        x_label_list = [str(i) for i in range(map.shape[1])]
        ax.set_xticklabels(x_label_list)
    else:
        ax.set_xticklabels(xstick)

    # the y-axis
    plt.ylabel(ylabel)
    ax.set_yticks(range(map.shape[0]))
    y_label_list = [str(i + 1) for i in range(map.shape[0])]
    ax.set_yticklabels(y_label_list)

    # for each cell
    for x_index in range(map.shape[1]):
        for y_index in range(map.shape[0]):
            label = None
            if type(map[y_index, x_index]) == np.bool_:
                label = str(map[y_index, x_index])
            else:
                label = str(np.round(map[y_index, x_index], 3))
            ax.text(x_index, y_index, label, color='black', ha='center', va='center')

    plt.grid()
    plt.colorbar()

    return fig
