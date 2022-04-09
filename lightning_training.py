
import torch
from torch import nn

import pytorch_lightning as pl


from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import BertModel
from transformers import BertTokenizer



from first_model.lightning_bert_nli import BertNliLight
from custom_data_set import SnliDataset



# data
train_data_set = SnliDataset(dir = "data/snli_1.0_train.txt" , nb_sentences= 1000 , msg = False)
test_data_set = SnliDataset(dir = "data/snli_1.0_test.txt" , nb_sentences = 100 , msg = False)

train_loader = DataLoader(train_data_set, batch_size=4)
val_loader = DataLoader(test_data_set, batch_size=4)

# model
model = BertNliLight()

# training
trainer = pl.Trainer()
trainer.fit(model, train_loader, val_loader)
    