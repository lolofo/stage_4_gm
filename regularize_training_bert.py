import argparse

# import self as self
import torch

import pytorch_lightning as pl
from datasets import load_dataset

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import nn

from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy

from transformers import BertModel
from transformers import BertTokenizer

import os
from os import path

from pytorch_lightning import callbacks as cb

from modules import transforms as t

tk = BertTokenizer.from_pretrained('bert-base-uncased')

#############
### model ###
#############

class BertNliRegu(pl.LightningModule):
    """ BertNliLight modèle (Bert for snli task)

    """

    def __init__(self, freeze_bert=False, criterion=nn.CrossEntropyLoss()):
        super().__init__()

        self.save_hyperparameters("reg_mul")

        # bert layer
        # the bert layer will return the layer will return the attention weights
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_attentions=True  # return the attention weights
                                              )

        self.bert_output = None

        # classifier head
        self.classifier = nn.Sequential(
            # fully connected layer
            nn.Linear(in_features=768, out_features=3),
        )

        # multiplier of the reg term
        # if this term is high >> high regularization
        self.reg_mul = 0

        self.train_acc = Accuracy(num_class=3)
        self.val_acc = Accuracy(num_class=3)
        self.test_acc = Accuracy(num_class=3)
        self.criterion = criterion

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        # don't save any tensor with gradient, conflict in multiprocessing
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        cls_token = output.last_hidden_state[:, 0, :].clone()

        # the logits are the weights before the softmax.
        logits = self.classifier(cls_token)

        return logits, output

    def configure_optimizers(self):
        '''
        define the optimizer for the training
        '''
        optimizer = AdamW(self.parameters(), lr=5e-5)

        return optimizer

    ######################
    ### training steps ###
    ######################

    # calculation of the regularization term
    def entropy_regu(
            outputs,
            input_ids):
        # calculate the entropia terms based on the outputs
        # outputs >> bert output form
        spe_ids = torch.tensor([0, 101, 102])
        # indexes of the specials tokens for the bert tokenizer

        # logical mask
        mask = torch.tensor(torch.logical_not(torch.isin(input_ids, spe_ids)),
                            dtype=int)

        den_log = torch.log(mask.sum(dim=1))  # how many dim tokens do we have
        # we repeat the mask along the different axes of the heads and the different
        mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, 12, 12, 1)

        # shape [batch, layer, head, T = 150, T=150]
        attention_tensor = torch.stack(outputs.attentions,
                                       dim=1)

        # attention score for each head of each layer
        # shape of [b, l, h, T = 150] we have a score for each sentences
        as_scores = attention_tensor.sum(dim=-1)

        # with the new dimensions we perform the mask >> remove of the special tokens
        as_scores = torch.mul(as_scores, mask)

        as_scores = as_scores / as_scores.sum(dim=-1).unsqueeze(3).repeat(1, 1, 1, 150)

        # calculation of the entropia
        # as_scores [b, l, h, T]
        etp_scores = -as_scores * torch.nan_to_num(torch.log(as_scores))
        etp_scores = etp_scores.sum(dim=-1)  # shape [b, l, h]

        res = etp_scores.sum() / (etp_scores.shape[0] * etp_scores.shape[1] * etp_scores.shape[2])

        return res

    # at the end of

    def training_step(self, train_batch, batch_idx):
        input_ids, attention_mask, labels = train_batch
        logits, outputs = self(input_ids, attention_mask)

        # calculation of the loss
        reg_term = self.entropy_regu(outputs=outputs,
                                     input_ids=input_ids)
        loss = self.criterion(logits, labels) + self.reg_mul * reg_term

        class_pred = torch.softmax(logits, dim=1)

        return {'loss': loss, 'preds': class_pred, 'target': labels, 'reg_term': reg_term}

    def training_step_end(self, output):
        self.train_acc(output['preds'], output['target'])
        self.log("train/loss", output['loss'], on_step=True, on_epoch=False, logger=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=False, logger=True)
        self.log("train/reg", output["reg_term"], on_step=True, on_epoch=False, logger=True)

    ########################
    ### validation steps ###
    ########################

    def validation_step(self, val_batch, batch_idx):
        return self.training_step(val_batch, batch_idx)

    def validation_step_end(self, output):
        self.val_acc(output['preds'], output['target'])
        self.log("val/loss", output['loss'], on_step=False, on_epoch=True, logger=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, logger=True)

    ##################
    ### test steps ###
    ##################
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self.forward(input_ids, attention_mask)

        # some tools for the end_validation
        class_pred = torch.softmax(logits, dim=1)
        return {'preds': class_pred, 'target': labels}

    def test_step_end(self, output):
        # TODO :
        self.test_acc(output['preds'], output['target'])
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, logger=True)

        self.log("hp/auc", output["auc"], on_step=True, on_epoch=True)


################
### the data ###
################

class SNLIDataModule(pl.LightningDataModule):
    """
    Data module (pytorch lightning) for the SNLI dataset

    Attributes :
    ------------
        cache : the location of the data on the disk
        batch_size : batch size for the training
        num_workers : number of cpu heart to use for using the data
        nb_data : number of data for the training
        t_add_sep : modules class to add the [SEP] token
        t_tokenize : modules class to return the attention_masks and input ids
        t_tensor : modules class to transform the input_ids and att_mask into tensors
    """

    def __init__(self, cache: str, batch_size=8, num_workers=0, nb_data=-1):
        super().__init__()
        self.cache = cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nb_data = nb_data

        self.t_add_sep = t.AddSepTransform()
        self.t_tokenize = t.BertTokenizeTransform(max_pad=150)
        self.t_tensor = t.CustomToTensor()

    def prepare_data(self):
        # called on 1 gpu
        # verify if the data exists already on the disk
        if not path.exists(path.join(self.cache, 'snli')):
            load_dataset('snli', cache_dir=self.cache)

    def setup(self, stage: str = None):
        """
        set_up function :

            - this function will prepare the data by setting the differents attributes self.train_set ... etc

        :param stage : are we preparing the data for the training part or the test part.
        """

        # called on every GPU
        # load dataset from cache in each instance of GPU
        if stage == 'fit' or stage is None:
            self.train_set = load_dataset('snli', split='train', cache_dir=self.cache).filter(
                lambda example: example['label'] >= 0)
            self.val_set = load_dataset('snli', split='validation', cache_dir=self.cache).filter(
                lambda example: example['label'] >= 0)

            if self.nb_data > 0:
                # else we take all the datas present in the dataset.
                self.train_set = self.train_set.shard(num_shards=len(self.train_set) // self.nb_data + 1, index=0)
                self.val_set = self.val_set.shard(num_shards=len(self.val_set) // self.nb_data + 1, index=0)

        if stage == 'test' or stage is None:
            self.test_set = load_dataset('snli', split='test', cache_dir=self.cache).filter(
                lambda example: example['label'] >= 0)
            if self.nb_data > 0:
                self.test_set = self.test_set.shard(num_shards=len(self.test_set) // self.nb_data + 1, index=0)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    ## ======= PRIVATE SECTIONS ======= ##
    def collate(self, batch):
        """
        collate function modify the structure of a Batch
        """
        batch = self.list2dict(batch)
        texts = self.t_add_sep(batch['premise'], batch['hypothesis'])
        input_ids, attention_mask = self.t_tokenize(texts)
        input_ids = self.t_tensor(input_ids)
        attention_mask = self.t_tensor(attention_mask)
        labels = self.t_tensor(batch['label'])
        return input_ids, attention_mask, labels

    def list2dict(self, batch):
        # convert list of dict to dict of list
        if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}  # handle case where no batch
        return {k: [row[k] for row in batch] for k in batch[0]}


def get_num_workers() -> int:
    '''
    Get maximum logical workers that a machine has
    Args:
        default (int): default value

    Returns:
        maximum workers number
    '''
    if hasattr(os, 'sched_getaffinity'):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass

    num_workers = os.cpu_count()
    return num_workers if num_workers is not None else 0
