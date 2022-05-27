"""
first test of loading a model and use the attention files
"""
from training_bert import BertNliLight
from custom_data_set import SnliDataset
from torch.utils.data import DataLoader

from attention_algorithms.raw_attention import create_adj_matrix
from attention_algorithms.raw_attention import create_attention_graph
from attention_algorithms.raw_attention import draw_attention_graph

from attention_algorithms.heads_role import attention_confidence
from attention_algorithms.heads_role import plot_confidence

import matplotlib.pyplot as plt
import os
from os import path

# first load the model
# we load the pre-trained model for the tests.
cache = path.join(os.getcwd(), '.cache', 'logs', 'bert', '0', 'checkpoints')
model = BertNliLight()
model.eval()

# load some data just load one sentence
data_set = SnliDataset(nb_sentences=1, msg=False)
data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

sentences, masks, train_labels = next(iter(data_loader))

attention_tensors, tokens, input_ids, attention_mask = model.get_attention(sentences, masks, test_mod=True)

adj_matrix, labels = create_adj_matrix(attention_tensors,
                                       heads_concat=False,
                                       num_head=1,
                                       test_mod=True)

###################################
### test of the attention graph ###
###################################

plots_folder = os.path.join('.cache', 'plots')
graph_folder = ""
if not path.exists(path.join(plots_folder, "attention_test_runs")):
    os.mkdir(path.join(path.join(plots_folder, "attention_test_runs")))

graph_folder = path.join(path.join(plots_folder, "attention_test_runs"))

g = create_attention_graph(attention_tensors, heads_concat=False, num_head=0)
g, fig = draw_attention_graph(g, labels, n_layers=12, tokens=tokens, graph_width=25)
plt.savefig(os.path.join(graph_folder, 'attention_graph_head_1.png'))

data_set = SnliDataset(nb_sentences=32, msg=False)
data_loader = DataLoader(data_set, batch_size=32, shuffle=False)

sentences, masks, train_labels = next(iter(data_loader))
map = attention_confidence(model, sentences, masks)

fig = plot_confidence(map)
plt.savefig(os.path.join(graph_folder, 'confidence_map.png'))
