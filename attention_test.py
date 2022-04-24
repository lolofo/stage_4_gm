"""
first test of loading a model and use the attention files
"""

import torch
from first_model.bert_nli import BertNli
from custom_data_set import SnliDataset
from torch.utils.data import DataLoader

from attention_algorithms.raw_attention import attention_tools

# first load the model
model = BertNli()
model.load_state_dict(torch.load("checkpoint/default.pt"))
model.eval()

# load some data just load one sentence
data_set = SnliDataset(nb_sentences=1, msg=False)
data_loader = DataLoader(data_set, batch_size=1, shuffle=True)

sentences, masks, train_labels = next(iter(data_loader))
attention_tensors, input_ids, attention_mask = model.get_attention(sentences, masks)

print(attention_tensors.shape)
print(input_ids.shape)
print(attention_mask.shape)

res, tokens = attention_tools(input_ids, attention_mask, attention_tensors)

print(tokens)
print(res.shape)

print(torch.sum(attention_mask))

print(res[0, 4, 9, 4, 5])

print(res[0, 8, 1, 7, 9])
print("end test")