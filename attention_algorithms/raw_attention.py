"""
the raw attention :
    - look at the attention weights
    - make some representation of the attention weigths
"""

import numpy as np
import torch
from transformers import BertTokenizer
import networkx as nx

# the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def attention_tools(input_ids,
                    attention_mask,
                    attention_tensor,
                    ):
    """
    input : attention_tensor -> torch tensor of the shape (batch_size = 1 , num_layers , num_heads , length , length)
            attention_mask   -> where are the padding tokens
            input_ids        -> ids of the different tokens (replace them with the original tokens)

    - this function will use the attention mask to delete the padding tokens
    - we will convert the input_ids into the different tokens

    then it will return :
            - the tokens
            - the torch tensor without the padding tokens
    """
    pass


def heads_transf(attention_tensor: torch.tensor,
                 heads_concat: bool = True,
                 num_head: int = -1,
                 ) -> torch.tensor:
    """
    input : attention_tensor -> torch tensor of the shape (batch_size = 1 , num_layers , num_heads , length , length)
            heads_concat     -> should we concat the heads
            num_heads        -> if we do not concat the heads what head should we take ?

    output : torch tensor of shape (num_layers , len , len)
    """
    res = None

    if heads_concat:
        # operation to concat the different heads
        pass
    else:
        res = attention_tensor[0, :, num_head, :, :]

    return res


def create_adj_matrix(attention_tensor: torch.tensor,
                      tokens,
                      heads_concat: bool = True,
                      num_head: int = -1
                      ):
    """
    we will create the adjacent matrix of the attention graph
    the objectif is to make the creation of the attention graph easier

    return the matrix to create a networkx graph
    return also a label for each node to know for each node in the graph to which token it corresponds
    """

    mat = heads_transf(attention_tensor, heads_concat, num_head)
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length))

    # the labels -> the name of each node to know were it is.
    labels = {}

    for i in range(n_layers):
        if i == 0:
            for u in range(length):
                # input labels
                buff = str(u) + "_" + tokens[u]
                labels[buff] = u
        else:
            for u in range(length):
                k_u = length * i + u
                buff = "L_" + str(i + 1) + "_" + str(u)
                labels[buff] = k_u
                for v in range(length):
                    k_v = length * (i - 1) + v
                    adj_mat[k_u][k_v] = mat[i][u][v]

    return adj_mat, labels


def create_attention_graph(attention_tensor: torch.tensor,
                           tokens,
                           heads_concat: bool = True,
                           num_head: int = -1
                           ):
    """
    creation of a network x Digraph based on the adj_matrix.
    """
    adj_mat, _ = create_adj_matrix(attention_tensor, tokens, heads_concat, num_head)
    buffer = adj_mat.detach().numpy()
    adj_mat = np.copy(buffer)

    # creation of the graph with the adjacency matrix.
    g = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph())

    return g
