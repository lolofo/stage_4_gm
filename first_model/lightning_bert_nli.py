import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from torch.optim import AdamW

from transformers import BertModel
from transformers import BertTokenizer

from torchmetrics.classification import Accuracy

criterion = nn.CrossEntropyLoss()

"""
TODO: 
    - create a function to access to the hidden attention state and make a good visualization of it.
"""


class BertNliLight(pl.LightningModule):

    def __init__(self, freeze_bert=False):
        super().__init__()

        # bert layer
        # the bert layer will return the layer will return the attention weights
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

        self.bert_output = None

        # classifier head
        self.classifier = nn.Sequential(
            # fully connected layer
            nn.Linear(in_features=768, out_features=3),

        )

        self.val_acc = Accuracy(num_class=3)
        self.train_acc = Accuracy(num_class=3)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        '''
        input_ids :      torch.tensor of shape (batch_size , max_pad)
        attention_mask : torch.tensor of shape (batch_size , max_pad)

        The output of the model will be the logits of the model (weights before softmax)
        '''

        self.bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)

        cls_token = self.bert_output.last_hidden_state[:, 0, :]

        # the logits are the weights before the softmax.
        logits = self.classifier(cls_token)

        return logits

    def configure_optimizers(self):
        '''
        define the optimizer for the training
        '''
        optimizer = AdamW(self.parameters(), lr=5e-5)

        return optimizer

    ######################
    ### training steps ###
    ######################

    def training_step(self, train_batch, batch_idx):
        input_ids, attention_mask, labels = train_batch
        logits = self.forward(input_ids, attention_mask)

        # calculation of the loss
        loss = criterion(logits, torch.max(labels, 1)[1])

        class_pred = torch.max(logits, 1)[1]
        class_true = torch.max(labels, 1)[1]

        self.train_acc(class_pred, class_true)

        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, logger=True)

        return loss

    ########################
    ### validation steps ###
    ########################

    def validation_step(self, val_batch, batch_idx):
        input_ids, attention_mask, labels = val_batch
        logits = self.forward(input_ids, attention_mask)

        # some tools for the end_validation
        class_pred = torch.max(logits, 1)[1]
        class_true = torch.max(labels, 1)[1]

        self.val_acc(class_pred, class_true)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, logger=True)
