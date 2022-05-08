import matplotlib.figure
import numpy as np
import torch
from transformers import BertTokenizer
import networkx as nx
import matplotlib.pyplot as plt

from typing import Union

# the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def heads_transf(attention_tensor: torch.tensor,
                 heads_concat: bool = True,
                 num_head: int = -1,
                 ) -> torch.tensor:
    """ Agregation of the different attention heads

    :param attention_tensor : torch.tensor of the shape (1, num_layers, num_head, length, length)
    :param heads_concat : if true we proceed head agregation (combination of the different heads)
    :param num_head : if we do not proceed any agregation, what head should we use

    :return: return a torch.tensor of the shape (num_layers , length , length)
    """

    res = None

    if heads_concat:
        # operation to concat the different heads
        pass
    else:
        res = attention_tensor[0, :, num_head, :, :]

    return res


def create_adj_matrix(attention_tensor: torch.tensor,
                      heads_concat: bool = True,
                      num_head: int = -1,
                      test_mod: bool = False,
                      ) -> Union[np.array, dict]:
    """ Creation of the adjacency matrix for the attention graph

    :param attention_tensor : torch.tensor of the shape(1 , num_layers, num_heads , length, length)
    :param tokens : list of the tokens in the sentence (for the labels)
    :param heads_concat : concatenation or not of the heads
    :param num_head : if we don't concat the heads, what head should we use
    :param test_mod : proceed some tests to be sure about what we are doing the test mod can only be used

    :return: - adj_matrix : numpy array of the shape (num_layers*length , num_layers*length)
             - labels : the labels of each node thus we can find them in the matrix
                     the keys of this dict are of the form Layer_1_CLS
    """

    mat = heads_transf(attention_tensor, heads_concat, num_head)
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros((n_layers * length, n_layers * length))

    # the labels -> the name of each node to know where it is.
    labels = {}

    for i in range(n_layers):
        if i == 0:
            for u in range(length):
                # input labels
                buff = "Layer_" + str(i + 1) + "_" + str(u)
                labels[buff] = u
        else:
            for u in range(length):
                k_u = length * i + u
                buff = "Layer_" + str(i + 1) + "_" + str(u)
                labels[buff] = k_u
                for v in range(length):
                    k_v = length * (i - 1) + v
                    adj_mat[k_u][k_v] = mat[i][u][v]
    # the test part
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
                            if attention_tensor[0, n, num_head, x, y].detach().numpy() != adj_mat[ \
                                    length * n + x, length * (n - 1) + y]:
                                passed = False

            if passed:
                print(u'\u2713')
            else:
                print("x")

    return adj_mat, labels


def create_attention_graph(attention_tensor: torch.tensor,
                           heads_concat: bool = True,
                           num_head: int = -1
                           ) -> nx.classes.digraph.DiGraph:
    """Creation of a networkx Digraph based on the adj_matrix.

    :param attention_tensor : torch tensor of the shape shape(1 , num_layers, num_heads , length, length)
    :param heads_concat : should we proceed some agregation between the different heads
    :param num_head : if we don't proceed any agregation what head should we choose

    :return:
    """
    adj_mat, _ = create_adj_matrix(attention_tensor, heads_concat, num_head)
    buffer = adj_mat
    adj_mat = np.copy(buffer)

    # creation of the graph with the adjacency matrix.
    g = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph())

    for i in np.arange(adj_mat.shape[0]):
        for j in np.arange(adj_mat.shape[1]):
            nx.set_edge_attributes(g, {(i, j): adj_mat[i, j]}, 'capacity')

    return g


def draw_attention_graph(g: nx.classes.digraph.DiGraph,
                         labels_to_index: dict,
                         tokens: list,
                         graph_width: int = 20,
                         n_layers: int = 12
                         ) -> Union[nx.classes.digraph.DiGraph, matplotlib.figure.Figure]:
    """ Draw the attention graph

    :param g : the attention graph (built on the attention matrix, but can also be built on the flow)
    :param labels_to_index : python dictionnary, the name of the different nodes
    :param tokens : the tokens, the sentence we are manipulating
    :param graph_width : the size of the final graph
    :param n_layers : number of multi-head-attention layer in the model (in bert it is 12)

    :return: the graph and the figure

    """
    pos = {}
    label_pos = {}
    length = len(tokens)
    for i in np.arange(n_layers):
        for k_f in np.arange(length):
            pos[i * length + k_f] = ((i + 0.4) * 2, length - k_f)
            label_pos[i * length + k_f] = (i * 2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = tokens[int(key.split("_")[-1])]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ''

    fig = plt.figure(figsize=(graph_width, graph_width))

    nx.draw_networkx_nodes(g, pos, node_color='green', labels=index_to_labels, node_size=50)
    nx.draw_networkx_labels(g, pos=label_pos, labels=index_to_labels, font_size=10)

    all_weights = []
    for (node1, node2, data) in g.edges(data=True):
        all_weights.append(data['weight'])

    unique_weights = list(set(all_weights))

    # 4 c. Plot the edges - one by one!
    for weight in unique_weights:
        weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in g.edges(data=True) if \
                          edge_attr['weight'] == weight]

        w = weight
        width = w
        nx.draw_networkx_edges(g, pos, edgelist=weighted_edges, width=width, edge_color='darkblue')

    return g, fig
