import matplotlib.figure
import numpy as np
import torch
from transformers import BertTokenizer
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from typing import Union

# the tokenizer
tk = BertTokenizer.from_pretrained('bert-base-uncased')


def hightlight_txt(tokens, attention, show_pad=False):
    """
    Build an HTML of text along its weights.
    Args:
        tokens: list of tokens
        attention: list of attention weights
        show_pad: whethere showing padding tokens
    """
    assert len(tokens) == len(attention), f'Length mismatch: f{len(tokens)} vs f{len(attention)}'

    MAX_ALPHA = 0.8  # transparency

    highlighted_text = ''
    w_min, w_max = torch.min(attention), torch.max(attention)

    # In case of uniform: highlight all text
    if w_min == w_max:
        w_min = 0.

    w_norm = (attention - w_min) / (w_max - w_min)
    w_norm = [w / MAX_ALPHA for w in w_norm]

    if not show_pad:
        id_non_pad = [i for i, tk in enumerate(tokens) if tk != '[pad]']
        w_norm = [w_norm[i] for i in id_non_pad]
        tokens = [tokens[i] for i in id_non_pad]

    highlighted_text = [f'<span style="background-color:rgba(135,206,250, {weight});">{text}</span>' for weight, text in
                        zip(w_norm, tokens)]

    return ' '.join(highlighted_text)


##############################################
## the raw attention class >> define the main function to manipulate the attention
## define it as a class will help us to regroup some technics
##############################################

class RawAttention:
    def __init__(self, model, input_ids, attention_mask, test_mod=True,
                 *args, **kwargs):
        # the base values
        self.input_ids = input_ids
        self.attention_mask = attention_mask

        # the corresponding tokens
        mask = attention_mask[0, :].detach().numpy() == 1
        self.tokens = tk.convert_ids_to_tokens(input_ids[0, mask])

        # the attention tensor
        self.attention_tensor = None
        outputs = model.bert(input_ids=input_ids,
                             attention_mask=attention_mask,
                             *args, **kwargs)

        buff = outputs.attentions
        res = torch.stack(buff, dim=1)

        # remove the padding tokens
        mask = attention_mask[0, :].detach().numpy() == 1
        res = res[:, :, :, mask, :]
        res = res[:, :, :, :, mask]
        self.attention_tensor = res.detach().clone() # >> fastest way to clone a tensor

        ## >> start test for the attention tensor
        if test_mod:
            print("test passed : ", end='')
            passed = True
            for n in range(len(buff)):
                for n_head in range(12):
                    for x in range(input_ids.shape[1]):
                        for y in range(input_ids.shape[1]):
                            if mask[x] == 1 and mask[y] == 1:
                                if buff[n][0, n_head, x, y] != self.attention_tensor[0, n, n_head, x, y]:
                                    passed = False
                                    break

            if passed:
                print(u'\u2713')
            else:
                print("x")
        ## >> end test

        self.att_tens_agr = None  # the tensor with heads agregation (which layer is significant)
        self.att_tens_lay_agr = None  # the tensor with layer agregation (which head is significant)

        # attention graph stuff
        self.adj_mat = None
        self.label = None
        self.attention_graph = None
        self.set_gr = False

    ########################
    ### heads agregation ###
    ########################

    def heads_agregation(self,
                         heads_concat: bool = True,
                         num_head: int = -1,
                         ):
        """ Agregation of the different attention heads

        :param heads_concat : if true we proceed head agregation (combination of the different heads)
        :param num_head : if we do not proceed any agregation, what head should we use
        """

        if heads_concat:
            # TODO : heads agregation
            pass
        else:
            # clone just the tensor without the gradient
            # here the gradient is not usefull
            self.att_tens_agr = self.attention_tensor[0, :, num_head, :, :].detach().clone()

    ################################
    ### defining the graph tools ###
    ################################

    def _create_adj_matrix(self,
                           heads_concat: bool = True,
                           num_head: int = -1,
                           test_mod: bool = False,
                           ):
        """ Creation of the adjacency matrix for the attention graph

        /!\ WARNING : the adj matrix has only sens for the heads agregation problem

        :param heads_concat : concatenation or not of the heads
        :param num_head : if we don't concat the heads, what head should we use
        :param test_mod : proceed some tests to be sure about what we are doing the test mod can only be used
        """

        self.heads_agregation(num_head=num_head, heads_concat=heads_concat)
        length = len(self.tokens)
        n_layers, _, _ = self.att_tens_agr.shape # number of attention heads
        adj_mat = np.zeros((n_layers * length, n_layers * length))

        # the labels -> the name of each node to know where it is.
        labels = {}

        for i in range(n_layers):
            if i == 0:
                for u in range(length):
                    # input labels
                    buff = "Layer_" + str(i) + "_" + str(u)
                    labels[buff] = u
            else:
                for u in range(length):
                    k_u = length * i + u
                    buff = "Layer_" + str(i) + "_" + str(u)
                    labels[buff] = k_u
                    for v in range(length):
                        k_v = length * (i - 1) + v
                        adj_mat[k_u][k_v] = self.att_tens_agr[i][u][v]

        # >> start the test
        if test_mod:
            if num_head < 0 or num_head > 11:
                print("error : please choose a head for the test part")
            else:
                print("test passed : ", end="")
                passed = True
                # proceed the test
                # we will see if the adjacency matrix contains the right values
                # thanks to this we will be able to test the previous function also
                for n in range(n_layers):
                    if n > 0:
                        for x in range(length):
                            for y in range(length):
                                if self.attention_tensor[0, n, num_head, x, y].detach().numpy() != adj_mat[ \
                                        length * n + x, length * (n - 1) + y]:
                                    passed = False
                                    break

                if passed:
                    print(u'\u2713')
                else:
                    print("x")
        # >> end the test

        # setting up the attributes
        self.adj_mat = np.copy(adj_mat)
        self.label = labels.copy()

    def _create_attention_graph(self):
        """Creation of a networkx Digraph based on the adj_matrix.

        /!\ >> the attention graph has sens only for the heads agregation.
        for the the layer agregation it has just no sens
        """

        # creation of the graph with the adjacency matrix.
        g = nx.from_numpy_matrix(self.adj_mat, create_using=nx.DiGraph())

        for i in np.arange(self.adj_mat.shape[0]):
            for j in np.arange(self.adj_mat.shape[1]):
                nx.set_edge_attributes(g, {(i, j): self.adj_mat[i, j]}, 'capacity')

        # the graph is also created to have capacities so we can perform max flow problem on it
        self.attention_graph = g.copy()

    def set_up_graph(self, num_head, heads_concat, test_mod=False):
        self._create_adj_matrix(heads_concat=heads_concat, num_head=num_head, test_mod=test_mod)
        self._create_attention_graph()

        # the graph is now set up !
        self.set_gr = True

    def draw_attention_graph(self,
                             graph_width: int = 20,
                             n_layers: int = 12
                             ) -> Union[nx.classes.digraph.DiGraph, matplotlib.figure.Figure]:
        """ Draw the attention graph

        :param graph_width : the size of the final graph
        :param n_layers : number of multi-head-attention layer in the model (in bert it is 12)
        """
        if not (self.set_gr):
            print("/!\ WARNINNG : you didn't set up the graph so we proceed it")
            # we must find a way to concat the different heads
            self.set_up_graph(heads_concat=True)

        pos = {}
        label_pos = {}
        length = len(self.tokens)
        for i in np.arange(n_layers):
            for k_f in np.arange(length):
                pos[i * length + k_f] = ((i + 0.4) * 2, length - k_f)
                label_pos[i * length + k_f] = (i * 2, length - k_f)

        index_to_labels = {}
        for key in self.label:
            index_to_labels[self.label[key]] = self.tokens[int(key.split("_")[-1])]
            if self.label[key] >= length:
                # only label the labels on the left side
                index_to_labels[self.label[key]] = ''

        fig = plt.figure(figsize=(graph_width, graph_width))

        #  label=index_to_labels
        nx.draw_networkx_nodes(self.attention_graph, pos, node_color='green', node_size=50)
        nx.draw_networkx_labels(self.attention_graph, pos=label_pos, labels=index_to_labels, font_size=10)

        # draw the edges
        all_weights = []
        for (node1, node2, data) in self.attention_graph.edges(data=True):
            all_weights.append(data['weight'])

        unique_weights = list(set(all_weights))

        for weight in unique_weights:
            weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in self.attention_graph.edges(data=True) if \
                              edge_attr['weight'] == weight]

            w = weight
            width = w
            nx.draw_networkx_edges(self.attention_graph, pos, edgelist=weighted_edges, width=width,
                                   edge_color='darkred')

        return fig
