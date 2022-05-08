"""
first test of loading a model and use the attention files
"""

import torch
from first_model.bert_nli import BertNli
from custom_data_set import SnliDataset
from torch.utils.data import DataLoader

from attention_algorithms.raw_attention import create_adj_matrix
from attention_algorithms.raw_attention import create_attention_graph
from attention_algorithms.raw_attention import draw_attention_graph
import matplotlib
import matplotlib.pyplot as plt

# first load the model
model = BertNli()
# model.load_state_dict(torch.load("checkpoint/default.pt"))
model.eval()

# load some data just load one sentence
data_set = SnliDataset(nb_sentences=1, msg=False)
data_loader = DataLoader(data_set, batch_size=1, shuffle=True)

sentences, masks, train_labels = next(iter(data_loader))

attention_tensors, tokens, input_ids, attention_mask = model.get_attention(sentences, masks, test_mod=True)

adj_matrix, labels = create_adj_matrix(attention_tensors,
                                       heads_concat=False,
                                       num_head=1,
                                       test_mod=True)

###################################
### test of the attention graph ###
###################################

g = create_attention_graph(attention_tensors, heads_concat=False, num_head=0)
g, fig = draw_attention_graph(g, labels, n_layers=12, tokens=tokens, graph_width=25)
plt.savefig("plots/attention_graph_head_1.png")

g = create_attention_graph(attention_tensors, heads_concat=False, num_head=1)
g, fig = draw_attention_graph(g, labels, n_layers=12, tokens=tokens, graph_width=25)
plt.savefig("plots/attention_graph_head_2.png")

g = create_attention_graph(attention_tensors, heads_concat=False, num_head=2)
g, fig = draw_attention_graph(g, labels, n_layers=12, tokens=tokens, graph_width=25)
plt.savefig("plots/attention_graph_head_3.png")

g = create_attention_graph(attention_tensors, heads_concat=False, num_head=10)
g, fig = draw_attention_graph(g, labels, n_layers=12, tokens=tokens, graph_width=25)
plt.savefig("plots/attention_graph_head_11.png")
