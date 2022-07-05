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

class BertNliLight(pl.LightningModule):
    """ BertNliLight modèle (Bert for snli task)

    """

    def __init__(self, freeze_bert=False, criterion=nn.CrossEntropyLoss()):
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

        self.train_acc = Accuracy(num_class=3)
        self.val_acc = Accuracy(num_class=3)
        self.test_acc = Accuracy(num_class=3)
        self.criterion = criterion

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        # don't save any tensor with gradient, conflict in multiprocessing
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        cls_token = output.last_hidden_state[:, 0, :]

        # the logits are the weights before the softmax.
        logits = self.classifier(cls_token)

        return logits

    # return the attention.
    def _get_att_weight(self, input_ids, attention_mask, *args, **kwargs):
        self.bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        result = torch.clone(self.bert_output.attentions)

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
        logits = self(input_ids, attention_mask)

        # calculation of the loss
        loss = self.criterion(logits, labels)

        class_pred = torch.softmax(logits, dim=1)

        return {'loss': loss, 'preds': class_pred, 'target': labels}

    def training_step_end(self, output):
        self.train_acc(output['preds'], output['target'])
        self.log("train/loss", output['loss'], on_step=False, on_epoch=True, logger=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, logger=True)

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
        self.test_acc(output['preds'], output['target'])
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, logger=True)


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # .cache folder >> the folder were everything will be saved
    cache = path.join(os.getcwd(), '.cache')
    if not path.exists(path.join(cache, 'plots')):
        os.mkdir(path.join(cache, 'plots'))

    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)

    # what model we should use the default is 1 >> the one created in this file
    parser.add_argument('-t', '--model_type', type=int, default=1)

    # default datadir >> ./.cache/dataset >> cache for our datamodule.
    parser.add_argument('-d', '--data_dir', default=path.join(cache, 'dataset'))

    # log_dir for the logger
    parser.add_argument('-s', '--log_dir', default=path.join(cache, 'logs'))

    parser.add_argument('-n', '--nb_data', type=int, default=-1)
    parser.add_argument('-mn', '--model_name')

    # config to distinguish experimentations
    parser.add_argument('--exp', action='store_true')  # mode experiment: avoid printing progress bars

    # save in [args.log_dir]/[experiments]/[version]
    parser.add_argument('--experiment', type=str, default='test')
    parser.add_argument('--version', type=str, default='0.0')

    # config for cluster distribution
    parser.add_argument('--num_workers', type=int,
                        default=get_num_workers())  # auto select appropriate cores in machine
    parser.add_argument('--accelerator', type=str, default='auto')  # auto select GPU if exists

    args = parser.parse_args()

    # Summary information
    print('>> workers: ', args.num_workers)
    print('>> nb_data: ', args.nb_data)

    dm = SNLIDataModule(
        cache=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nb_data=args.nb_data
    )

    model = None
    if args.model_type == 1:
        model = BertNliLight(criterion=nn.CrossEntropyLoss())

    ######################
    ### trainer config ###
    ######################

    # set the direction to visualize the logs of the training
    # the visualization will be done with tensorboard.
    logger = TensorBoardLogger(
        save_dir=args.log_dir,  # the main log folder
        name=args.experiment,  # name of the log >> related to the name of the model we use
        version=args.version,  # version of the log
        default_hp_metric=False  # deactivate hp_metric on tensorboard visualization
    )
    # logger = TensorBoardLogger(name=args.log_dir, save_dir=log_dir + '/')

    # call back
    early_stopping = cb.EarlyStopping('val/loss', patience=5, verbose=args.exp,
                                      mode='min')  # stop if no improvement withing 5 epochs
    model_checkpoint = cb.ModelCheckpoint(
        filename='best', monitor='val/loss', mode='min',  # save the minimum val_loss
    )

    trainer = pl.Trainer(
        max_epochs=args.epoch,
        accelerator=args.accelerator,  # auto use gpu
        enable_progress_bar=not args.exp,  # hide progress bar in experimentation
        log_every_n_steps=1,
        default_root_dir=args.log_dir,
        logger=logger,
        callbacks=[early_stopping, model_checkpoint],
        detect_anomaly=not args.exp
    )

    #############################
    ### training of the model ###
    #############################
    dm.setup(stage='fit')
    trainer.fit(model, datamodule=dm)

    dm.setup(stage='test')
    performance = trainer.test(
        ckpt_path='best',
        datamodule=dm
    )

    logger.log_metrics(performance[0])

    print('Training finished')
