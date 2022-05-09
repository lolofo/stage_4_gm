"""Heads role

Objectif :
    1 - performe the confidence of a head a see which head has something interesting to give :
        (base on this article : https://aclanthology.org/P19-1580.pdf)

"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def attention_confidence(model,
                         input_ids,
                         attention_mask,
                         n_layers: int = 12,
                         n_head: int = 12
                         ):
    """Calculation of the confidence of the different heads

    :param model : one of the model created in this repository
    :param input_ids : torch.tensor of the shape (batch_size , sentence_length)
    :param attention_mask : torch.tensor of the shape (batch_size , sentence_length)
    :param n_layers : number of layers in the model we are using
    :param n_head : number of attention head in the model

    :return: the treillis of the confidence values
    """
    nb_sentences = input_ids.shape[0]
    confidence_map = np.zeros((n_layers, n_head))  # 12 layers , 12 heads

    for i in range(nb_sentences):
        # proceed sentence by sentence because every sentence has a different length
        ids, msk = torch.tensor([input_ids[i, :].detach().numpy()]), \
                   torch.tensor([attention_mask[i, :].detach().numpy()])

        attention_tensor, tokens, _, _ = model.get_attention(ids, msk)
        nb_tokens = attention_mask[i, :].detach().numpy().sum()
        for n in range(n_layers):
            for h in range(n_head):
                buff = attention_tensor[0, n, h, 0:(nb_tokens - 1), :]
                buff = buff[:, 0:(nb_tokens - 1)].detach().numpy()
                confidence_map[n, h] += buff.max()

    return confidence_map / nb_sentences


def plot_confidence(map):
    fig = plt.figure(figsize=(10, 10))

    plt.imshow(map, aspect='auto', cmap='Blues')
    txt = "Confidence values"
    plt.title(txt)
    y_label_list = [str(i) for i in range(map.shape[0])]
    x_label_list = [str(i) for i in range(map.shape[1])]
    plt.xlabel('Head')
    plt.ylabel('Layer')
    ax = plt.gca()

    ax.set_xticks(range(map.shape[1]))
    ax.set_yticks(range(map.shape[0]))

    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)

    for x_index in range(map.shape[1]):
        for y_index in range(map.shape[0]):
            label = str(np.round(map[y_index, x_index], 3))
            plt.text(x_index, y_index, label, color='black', ha='center', va='center')

    # don't show the grid
    plt.grid()
    plt.colorbar()

    return fig

