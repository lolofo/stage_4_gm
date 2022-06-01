"""Heads role

Objectif :
    1 - performe the confidence of a head a see which head has something interesting to give :
        (base on this article : https://aclanthology.org/P19-1580.pdf)

"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from attention_algorithms.raw_attention import RawAttention
import seaborn as sns
sns.set_theme()


class HeadsRole:

    def __init__(self, input_ids, attention_masks,
                 n_layer: int = 12,
                 n_head: int = 12
                 ):

        self.input_ids = input_ids.detach().clone()
        self.attention_mask = attention_masks.detach().clone()
        self.nb_sentences = input_ids.shape[0]  # >> number of sentences of which we will do the training

        # the confidence map
        self.confidence_map = np.zeros((n_layer, n_head))  # 12 layers , 12 heads

    def attention_confidence(self,
                             model,
                             n_layers: int = 12,
                             n_head: int = 12
                             ):
        """Calculation of the confidence of the different heads

        :param model : model for the snli task
        :param n_layers : number of layers in the model we are using
        :param n_head : number of attention head in the model
        """

        for i in range(self.nb_sentences):
            # proceed sentence by sentence because every sentence has a different length
            ids, msk = torch.tensor([self.input_ids[i, :].detach().numpy()]), \
                       torch.tensor([self.attention_mask[i, :].detach().numpy()])

            raw_attention_buff = RawAttention(model,
                                              input_ids=ids,
                                              attention_mask=msk,
                                              test_mod=False)

            nb_tokens = len(raw_attention_buff.tokens)
            for n in range(n_layers):
                for h in range(n_head):
                    buff = raw_attention_buff.attention_tensor[0, n, h, 0:(nb_tokens - 1), :]
                    buff = buff[:, 0:(nb_tokens - 1)].detach().numpy()
                    self.confidence_map[n, h] += buff.max()

        self.confidence_map /= self.nb_sentences

    # plot the confidence map in the form of an image
    def plot_confidence(self):

        fig = plt.figure(figsize=(10, 10))

        plt.imshow(self.confidence_map, aspect='auto', cmap='Greens')
        txt = "Confidence values"
        plt.title(txt)
        y_label_list = [str(i) for i in range(self.confidence_map.shape[0])]
        x_label_list = [str(i) for i in range(self.confidence_map.shape[1])]
        plt.xlabel('Head')
        plt.ylabel('Layer')
        ax = plt.gca()

        ax.set_xticks(range(self.confidence_map.shape[1]))
        ax.set_yticks(range(self.confidence_map.shape[0]))

        ax.set_xticklabels(x_label_list)
        ax.set_yticklabels(y_label_list)

        for x_index in range(self.confidence_map.shape[1]):
            for y_index in range(self.confidence_map.shape[0]):
                label = str(np.round(self.confidence_map[y_index, x_index], 3))
                plt.text(x_index, y_index, label, color='black', ha='center', va='center')

        # don't show the grid
        plt.grid()
        plt.colorbar()

        return fig
