"""
Implementation of how to plot the ROC curve
"""
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from attention_algorithms.raw_attention import RawAttention


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


def combine_roc_ruves(Y_test, probs,
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


def plot_color_map(raw_attention_inst: RawAttention,
                   e_snli_annotation: list,
                   agr_type: str = "max"):
    # first >> agregation of the different heads
    raw_attention_inst.heads_agregation(heads_concat=True,
                                        agr_type=agr_type,
                                        num_head=-1)
    n_layer, n_tokens, _ = raw_attention_inst.att_tens_agr.shape
    map = np.zeros((n_tokens, n_layer + 1))

    for j in range(n_layer):
        buff = raw_attention_inst.att_tens_agr.detach().clone()
        for i in range(n_tokens):
            map[i, j] = buff[j, i, :].detach().numpy().max()

    map[:, n_layer] = e_snli_annotation

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(map, aspect='auto', cmap='Greens')
    plt.title("maximum from the previous layer // against the e_snli annotation")
    plt.xlabel('Layer')
    plt.ylabel('Tokens')
    ax = plt.gca()
    y_label_list = raw_attention_inst.tokens
    x_label_list = [str(i) for i in range(map.shape[1]-1)]+['e_snli']

    ax.set_xticks(range(map.shape[1]))
    ax.set_yticks(range(map.shape[0]))

    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)

    plt.grid()
    plt.colorbar()

    return fig
