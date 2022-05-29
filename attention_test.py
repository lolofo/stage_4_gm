"""
RUNNING SOME TESTS :
    - the plots will be on the folder plots/attention_test_runs in the .cache folder
"""
from training_bert import BertNliLight
from custom_data_set import SnliDataset
from torch.utils.data import DataLoader
from attention_algorithms.raw_attention import RawAttention
from attention_algorithms.heads_role import HeadsRole

import matplotlib.pyplot as plt
import os
from os import path

plots_folder = os.path.join('.cache', 'plots')
graph_folder = ""
if not path.exists(path.join(plots_folder, "attention_test_runs")):
    os.mkdir(path.join(path.join(plots_folder, "attention_test_runs")))

model = BertNliLight()
model.eval()

# load some data just load one sentence
data_set = SnliDataset(nb_sentences=1, msg=False)
data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

sentences, masks, train_labels = next(iter(data_loader))

print(f"shape of the sentences : {sentences.shape}")
print(f"shape of the masks : {masks.shape}")
print(f"numer of non-mask tokens : {masks.detach().numpy().sum()}")

raw_attention_inst = RawAttention(model=model,
                                  input_ids=sentences,
                                  attention_mask=masks)

print(f"shape of the attention tensor : {raw_attention_inst.attention_tensor.shape}")

raw_attention_inst.set_up_graph(num_head=1, heads_concat=False, test_mod=True)
fig = raw_attention_inst.draw_attention_graph()
plt.savefig(os.path.join(plots_folder, 'attention_test_runs', 'test_graph_head_1.png'))


# load a whole batch
data_set = SnliDataset(nb_sentences=1, msg=False)
data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

sentences, masks, train_labels = next(iter(data_loader))

heads_role_inst = HeadsRole(sentences, masks)
heads_role_inst.attention_confidence(model)
fig = heads_role_inst.plot_confidence()
plt.savefig(os.path.join(plots_folder, 'attention_test_runs', 'test_confidence_map.png'))

