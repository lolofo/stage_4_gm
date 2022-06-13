"""Heads role

Objectif :
    1 - performe the confidence of a head a see which head has something interesting to give :
        (base on this article : https://aclanthology.org/P19-1580.pdf)

"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from attention_algorithms.raw_attention import RawAttention
from tqdm import tqdm
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

        # the heads we will keep during this operation
        self.real_heads = np.zeros((n_layer, n_head))

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
        for _, i in enumerate(tqdm.tqdm(range(self.nb_sentences))):
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

    def heads_selection(self,
                        tr: float = 0.5):

        """ Selection of the attention heads
        :param tr: the treshold to keep the heads

        WARNING : the confidence map must be calculated on many sentences to be significant.
        """
        n_layer, n_head = self.confidence_map.shape
        for i in range(n_layer):
            for j in range(n_head):
                if self.confidence_map[i, j] >= tr:
                    self.real_heads[i, j] = 1

    # plot the confidence map in the form of an image
    def plot_confidence(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # plot the confidence map
        ax1.imshow(self.confidence_map, aspect='auto', cmap='Greens')
        y_label_list = [str(i) for i in range(self.confidence_map.shape[0])]
        x_label_list = [str(i) for i in range(self.confidence_map.shape[1])]
        ax1.set(xlabel='Head', ylabel='Layer')
        ax1.set_xticks(range(self.confidence_map.shape[1]))
        ax1.set_yticks(range(self.confidence_map.shape[0]))

        ax1.set_xticklabels(x_label_list)
        ax1.set_yticklabels(y_label_list)

        ax1.grid()

        for x_index in range(self.confidence_map.shape[1]):
            for y_index in range(self.confidence_map.shape[0]):
                label = str(np.round(self.confidence_map[y_index, x_index], 3))
                ax1.text(x_index, y_index, label, color='black', ha='center', va='center')

        ax1.set_title("The confidence values")

        # plot the pruned heads.
        ax2.imshow(self.real_heads, aspect='auto', cmap='Greens')
        y_label_list = [str(i) for i in range(self.real_heads.shape[0])]
        x_label_list = [str(i) for i in range(self.real_heads.shape[1])]
        ax2.set(xlabel='Head', ylabel='Layer')
        ax2.set_xticks(range(self.confidence_map.shape[1]))
        ax2.set_yticks(range(self.confidence_map.shape[0]))

        ax2.set_xticklabels(x_label_list)
        ax2.set_yticklabels(y_label_list)

        ax2.set_title("Pruned Heads")
        ax2.grid()

        return fig
