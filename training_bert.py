import argparse

import self as self
import torch

import pytorch_lightning as pl
from datasets import load_dataset

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import nn

from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy

from transformers import BertModel

import os
from os import path

from pytorch_lightning import callbacks as cb

from modules import transforms as t
import torchtext.transforms as T


#############
### model ###
#############

class BertNliLight(pl.LightningModule):

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
		'''
		input_ids :      torch.tensor of shape (batch_size , max_pad)
		attention_mask : torch.tensor of shape (batch_size , max_pad)

		The output of the model will be the logits of the model (weights before softmax)
		'''
		
		# don't save any tensor with gradient, conflict in multiprocessing
		output = self.bert(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
		cls_token = output.last_hidden_state[:, 0, :].clone()

		

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
		self.log("train_loss", output['loss'], on_step=False, on_epoch=True, logger=True)
		self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, logger=True)
		
	########################
	### validation steps ###
	########################

	def validation_step(self, val_batch, batch_idx):
		return self.training_step(val_batch, batch_idx)

	def validation_step_end(self, output):
		self.val_acc(output['preds'], output['target'])
		self.log("val_loss", output['loss'], on_step=False, on_epoch=True, logger=True)
		self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, logger=True)

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
		self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, logger=True)

    # return the attention.
	def get_attention(self,
                      input_ids,
                      attention_mask,
                      test_mod: bool = False,
                      *args, **kwargs):

	    outputs = self.bert(input_ids=input_ids,
	                        attention_mask=attention_mask,
	                        *args, **kwargs)
	
	    attention_tensor = outputs.attentions
	
	    res = torch.stack(attention_tensor, dim=1)
	
	    # remove the padding tokens
	
	    mask = attention_mask[0, :].detach().numpy() == 1
	    res = res[:, :, :, mask, :]
	    res = res[:, :, :, :, mask]
	
	    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, mask])
	
	    if test_mod:
	        print("test passed : ", end='')
	        passed = True
	        for n in range(len(attention_tensor)):
	            for n_head in range(12):
	                for x in range(input_ids.shape[1]):
	                    for y in range(input_ids.shape[1]):
	                        if mask[x] == 1 and mask[y] == 1:
	                            if attention_tensor[n][0, n_head, x, y] != res[0, n, n_head, x, y]:
	                                passed = False
	
	        if passed:
	            print(u'\u2713')
	        else:
	            print("x")
	
	    return res, tokens, input_ids, attention_mask


################
### the data ###
################

class SNLIDataModule(pl.LightningDataModule):

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

        # download dataset if not exist
        if not path.exists(path.join(self.cache, 'snli')):
            load_dataset('snli', cache_dir=self.cache)

    def setup(self, stage: str = None):
        # called on every GPU
        # load dataset from cache in each instance of GPU
        if stage == 'fit' or stage is None:
            self.train_set = load_dataset('snli', split='train', cache_dir=self.cache).filter(
                lambda example: example['label'] >= 0)
            self.val_set = load_dataset('snli', split='validation', cache_dir=self.cache).filter(
                lambda example: example['label'] >= 0)

            if self.nb_data > 0:
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
        # prepare batch of data for dataloader
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
	
	cache = path.join(os.getcwd(), '.cache')
	
	parser.add_argument('-e', '--epoch', type=int, default=1)
	parser.add_argument('-b', '--batch_size', type=int, default=4)
	parser.add_argument('-t', '--model_type', type=int, default=1)
	parser.add_argument('-d', '--data_dir', default=path.join(cache, 'dataset'))
	parser.add_argument('-s', '--log_dir', default=path.join(cache, 'logs'))
	parser.add_argument('-n', '--nb_data', type=int, default=-1)
	parser.add_argument('-mn', '--model_name')
	
	# config to distinguish experimentations
	parser.add_argument('--exp', action='store_true')  # mode experiment: avoid printing progress bars
	# save in [args.log_dir]/[experiments]/[version]
	parser.add_argument('--experiment', type=str, default='test')
	parser.add_argument('--version', type=str, default='0.0')
	
	# config for cluster distribution
	parser.add_argument('--num_workers', type=int, default=get_num_workers())  # auto select appropriate cores in machine
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
	
	'''
	TODO:
		- make some research to understand the parameters of the trainer
		- how to do cpu//gpu training
		- how to get the information of the training (done we do it with the tensorboard)
	'''
	
	# set the direction to visualize the logs of the training
	# the visualization will be done with tensorboard.
	logger = TensorBoardLogger(
		save_dir=args.log_dir,
		name=args.experiment,
		version=args.version,
		default_hp_metric=False  # deactivate hp_metric on tensorboard visualization
	)
	# logger = TensorBoardLogger(name=args.log_dir, save_dir=log_dir + '/')
	
	# call back
	early_stopping = cb.EarlyStopping('val_acc', patience=5, verbose=args.exp, mode='min')  # stop if no improvement withing 5 epochs
	model_checkpoint = cb.ModelCheckpoint(
		filename='best', monitor='val_loss', mode='min',  # save the minimum val_loss
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
	
	# print(trainer.logged_metrics) # Duc: Je ne pense pas necessaire
	
	dm.setup(stage='test')
	performance = trainer.test(
		ckpt_path='best',
		datamodule=dm
	)
	
	logger.log_metrics(performance[0])
	
	print('Training finished')
