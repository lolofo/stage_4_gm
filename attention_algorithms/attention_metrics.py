"""
Implementation of a ROC curve for our problem
"""
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import numpy as np

# --> plot the roc_curve of the model
def plot_roc_vurve(Y_test, probs):
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


def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


# --> plot the confusion matrix
def visu_conf_mat(Y_test, probs, tr):
    preds = probs

    print("the thresholds selected : {}".format(tr))
    # some metrics
    pred_label = np.array([1 if x > tr else 0 for x in preds])
    error_vector = np.sum(np.abs(pred_label - Y_test)) / len(Y_test)
    print("accuracy calculated : {}".format(np.round(1 - error_vector, 4)))
    print("number of yes labels predicted : {}".format(sum(pred_label)))
    print("number of yes labels : {}".format(sum(Y_test)))

    conf_mat = confusion_matrix(Y_test, pred_label)

    fig = plt.figure(figsize=(10, 10))

    plt.imshow(conf_mat, aspect='auto', cmap='Blues')
    txt = "confusion matrix"
    plt.title(txt)
    x_label_list = ["no_predicted", "yes_predicted"]
    y_label_list = ["true_no", "true_yes"]
    plt.xlabel('Predictions')
    plt.ylabel('Real values')
    ax = plt.gca()

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))

    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)

    # --> text on the matrix

    rates = np.array([["tnr", "fpr"], ["fnr", "tpr"]])
    N = len(Y_test)
    pos = sum(Y_test)
    neg = N - pos

    deno = np.array([[neg, neg], [pos, pos]])

    for x_index in range(2):
        for y_index in range(2):
            label = str(conf_mat[y_index, x_index]) + "\n"
            label += str(rates[y_index, x_index]) + " : \n"
            label += str(np.round(conf_mat[y_index, x_index] / deno[y_index, x_index], 3))
            plt.text(x_index, y_index, label, color='black', ha='center', va='center')

            if x_index == y_index:
                highlight_cell(x_index, y_index, color="red", linewidth=2)

    # don't show the grid
    plt.grid()
    plt.colorbar()

    return fig
