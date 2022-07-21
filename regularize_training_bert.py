import argparse

# import self as self
import json

import torch

import pytorch_lightning as pl
from e_snli_dataset import EsnliDataSet, MAX_PAD

# download the data
from dataset.esnli.e_snli_tok import process_e_snli_data
from data_download import download_e_snli_raw

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import nn

from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
from torchmetrics import AUROC

from transformers import BertModel
from transformers import BertTokenizer

import os
from os import path

from pytorch_lightning import callbacks as cb

from modules import transforms as t
from modules.logger import log, init_logging

tk = BertTokenizer.from_pretrained('bert-base-uncased')

#############
### model ###
#############

# constants for numerical stabilities
EPS = 1e-10
INF = 1e30


def L2D(batch):
    # convert list of dict to dict of list
    if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}
    return {k: [row[k] for row in batch] for k in batch[0]}


class BertNliRegu(pl.LightningModule):
    """ BertNliLight modèle (Bert for snli task)

    """

    def __init__(self, freeze_bert=False, criterion=nn.CrossEntropyLoss(),
                 reg_mul=0,
                 reg_lay: int = -1,
                 lr=5e-5,
                 exp: bool = False):
        super().__init__()
        self.exp = exp
        self.save_hyperparameters("reg_mul", "reg_lay", ignore=["exp"])

        self.lr = lr

        # bert layer
        # the bert layer will return the layer will return the attention weights
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_attentions=True  # return the attention weights
                                              )

        # classifier head
        self.classifier = nn.Sequential(
            # fully connected layer
            nn.Linear(in_features=768, out_features=3),
        )

        # multiplier of the reg term
        # if this term is high >> high regularization
        self.reg_mul = reg_mul
        self.reg_lay = reg_lay

        ### METRICS ###
        self.train_acc = Accuracy(num_class=3)
        self.val_acc = Accuracy(num_class=3)
        self.test_acc = Accuracy(num_class=3)
        self.criterion = criterion

        # auc plausibility scores
        self.train_auc = AUROC(pos_label=1)
        self.val_auc = AUROC(pos_label=1)
        self.test_auc = AUROC(pos_label=1)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        # don't save any tensor with gradient, conflict in multiprocessing
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        cls_token = output.last_hidden_state[:, 0, :]

        # the logits are the weights before the softmax.
        logits = self.classifier(cls_token)

        return {"logits": logits, "outputs": output}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer

    ######################
    ### training steps ###
    ######################
    def layer_reg(self,
                  outputs,
                  input_ids,
                  layer: int = 2):
        """Regularize the network layer by layer
        """
        # the mask , mask == 1 <==> special token
        spe_ids = torch.tensor([0, 101, 102]).to(self.device)
        mask = torch.isin(input_ids, spe_ids).type(torch.uint8).to(self.device)
        mask = mask.unsqueeze(1).repeat(1, 12, 1)
        # the attention
        attention_tensor = outputs.attentions[
            layer
        ]  # --> select the correct layer shape [batch, heads, MAX_PAD, MAX_PAD]
        as_scores = attention_tensor.sum(
            dim=len(attention_tensor.shape) - 2
        )
        # --> sum over the lines to have a distribution
        as_scores = torch.softmax(as_scores - INF * mask, dim=-1)  # get the distribution
        etp_scores = - as_scores * torch.log(as_scores + EPS * mask + 1e-16)
        etp_scores = etp_scores.sum(dim=-1)
        pen = etp_scores.mean() # mean over all the heads and the batch

        # --> for the AUC calculus
        sum_agreg = attention_tensor[:, :, :, :].sum(dim=1).sum(dim=1)

        # replace the specials tokens by zero
        sum_agreg = torch.where(torch.logical_not(torch.isin(input_ids, spe_ids)), sum_agreg, 0)

        buff = sum_agreg.clone()
        buff = torch.where(torch.logical_not(torch.isin(input_ids, spe_ids)), buff, 1e30)

        mins = buff.min(dim=-1)[0].unsqueeze(1).repeat(1, 150)
        maxs = sum_agreg.max(dim=-1)[0].unsqueeze(1).repeat(1, 150)

        sum_agreg = (sum_agreg - mins) / (maxs - mins)

        return {"pen": pen,
                "scores": (buff - mins) / (maxs - mins)}
        

    # calculation of the regularization term
    def model_regu(self,
                   outputs,
                   input_ids):
        # create the mask --> mask = 1 <=> special token
        spe_ids = torch.tensor([0, 101, 102]).to(self.device)
        mask = torch.isin(input_ids, spe_ids).type(torch.uint8).to(self.device)
        mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, 12, 12, 1)  # for the mask we have all the layers.
        # cerate the attention map
        attention_tensor = torch.stack(outputs.attentions, dim=1)

        as_scores = attention_tensor.sum(
            dim=len(attention_tensor.shape) - 2
        )

        # the entropia calculus
        as_scores = torch.softmax(as_scores - INF * mask, dim=-1)
        etp_scores = - as_scores * torch.log(as_scores + EPS * mask + 1e-16)
        etp_scores = etp_scores.sum(dim=-1)  # shape [b, l, h]
        pen = etp_scores.mean() # mean over all the heads and the layers (and the batch).

        # for the AUC calculus compute the heads agregation

        sum_agreg = attention_tensor[:, :, :, :, :].sum(dim=1).sum(dim=1).sum(dim=1)
        # replace the specials tokens by zero
        sum_agreg = torch.where(torch.logical_not(torch.isin(input_ids, spe_ids)), sum_agreg, 0)

        buff = sum_agreg.clone()
        buff = torch.where(torch.logical_not(torch.isin(input_ids, spe_ids)), buff, 1e30)

        mins = buff.min(dim=-1)[0].unsqueeze(1).repeat(1, 150)
        maxs = sum_agreg.max(dim=-1)[0].unsqueeze(1).repeat(1, 150)

        sum_agreg = (sum_agreg - mins) / (maxs - mins)
        return {"pen": pen,
                "scores": sum_agreg}

    # at the end of

    def on_train_start(self):
        init_hp_metrics = {'hp_/acc': 0, 'hp_/auc': 0}
        self.logger.log_hyperparams(self.hparams, init_hp_metrics)

    def training_step(self, train_batch, batch_idx):
        input_ids = train_batch["input_ids"]
        attention_mask = train_batch["attention_masks"]
        labels = train_batch["labels"]

        buff = self.forward(input_ids, attention_mask)
        logits = buff["logits"]
        outputs = buff["outputs"]

        loss = self.criterion(logits, labels)  # the loss based on the criterion
        reg_term = None
        if self.reg_lay > 0:
            # we regularize one given layer
            reg_term = self.layer_reg(outputs=outputs,
                                      input_ids=input_ids,
                                      layer=self.reg_lay)
        else:
            # we regularize all the layers
            reg_term = self.model_regu(outputs=outputs,
                                       input_ids=input_ids)

        loss += self.reg_mul * reg_term["pen"]

        class_pred = torch.softmax(logits, dim=1)
        # calculus of the attention score for the auc --> see the evolution of the auc through the epochs

        return {'loss': loss, 'preds': class_pred, 'target': labels, 'reg_term': reg_term["pen"],
                'auc': (torch.flatten(reg_term["scores"]).clone().detach(),
                        torch.flatten(train_batch["annotations"])
                        )
                }

    def training_step_end(self, output):
        # update the metrics
        self.train_acc(output['preds'], output['target'])
        self.train_auc(output["auc"][0], output["auc"][1])
        # add the metrics on the tensorboard
        self.log("train_/loss", output['loss'], on_step=True, on_epoch=False, logger=True)
        self.log("train_/acc", self.train_acc, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.log("train_/reg", self.reg_mul * output["reg_term"], on_step=True, on_epoch=False, logger=True,
                 prog_bar=True)
        self.log("train_/auc", self.train_auc, on_step=True, on_epoch=False, logger=True)

    ########################
    ### validation steps ###
    ########################
    # for the validation we must have leaves tensor -> no grad
    def validation_step(self, val_batch, batch_idx):
        return self.training_step(val_batch, batch_idx)

    def validation_step_end(self, output):
        # calculation of the metrics
        self.val_acc(output['preds'], output['target'])
        self.val_auc(output["auc"][0], output["auc"][1])
        # add the metrics on the tensorboard
        self.log("val_/loss", output['loss'], on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("val_/acc", self.val_acc, on_step=True, on_epoch=False, logger=True, prog_bar=False)
        self.log("val_/auc", self.val_auc, on_step=True, on_epoch=False, logger=True, prog_bar=False)

    #################################
    ###### END OF THE TRAIN EPOCH ###
    #################################

    def end_epoch(self, stage):
        d = dict()
        # don't show all the logs in non-exp mod
        # save some memory for the logs
        d[f"{stage}_acc"] = round(eval(f"self.{stage}_acc").compute().item(), 4)
        d[f"{stage}_auc"] = round(eval(f"self.{stage}_auc").compute().item(), 4)
        log.info(f"Epoch : {self.current_epoch} >> {stage}_metrics >> {d}")

    def on_validation_epoch_end(self):
        return self.end_epoch(stage="val")

    def on_train_epoch_end(self):
        return self.end_epoch(stage="train")

    ##################
    ### test steps ###
    ##################
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_masks"]
        labels = batch["labels"]

        buff = self.forward(input_ids, attention_mask)
        logits = buff["logits"]
        reg_term = self.model_regu(outputs=buff["outputs"],
                                   input_ids=input_ids)

        # some tools for the end_validation
        class_pred = torch.softmax(logits, dim=1)
        return {'preds': class_pred, 'target': labels,
                'auc': (
                    torch.flatten(reg_term["scores"]).clone().detach(),
                    torch.flatten(batch["annotations"])
                )
                }

    def test_step_end(self, output):
        """ The test step
        - for the test we only need the accuracy and the auc.
        """
        # the different metrics
        self.test_acc(output['preds'], output['target'])
        self.test_auc(output["auc"][0], output["auc"][1])
        # add the metrics on the tensorboard
        self.log("hp_/acc", self.test_acc, on_step=False, on_epoch=True, logger=True)
        self.log("hp_/auc", self.test_auc, on_step=False, on_epoch=True, logger=True)


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

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        log.info("we process the data download it can be a bit long ...")
        download_e_snli_raw(self.cache)
        process_e_snli_data(self.cache)

    def setup(self, stage: str = None):
        """
        set_up function :

            - this function will prepare the data by setting the differents attributes self.train_set ... etc

        :param stage : are we preparing the data for the training part or the test part.
        """

        # called on every GPU
        # load dataset from cache in each instance of GPU
        # TODO create the e-snli dataset
        if stage == 'fit' or stage is None:
            buff = None
            if self.nb_data > 0:
                buff = EsnliDataSet(split="TRAIN", nb_data=self.nb_data,
                                    cache_path=os.path.join(self.cache, "cleaned_data"))
            else:
                buff = EsnliDataSet(split="TRAIN", nb_data=-1,
                                    cache_path=os.path.join(self.cache, "cleaned_data"))
            # 80% train 20% validation
            train_size = int(0.8 * len(buff))
            val_size = len(buff) - train_size
            self.train_set, self.val_set = torch.utils.data.random_split(buff, [train_size, val_size])

        if stage == 'test' or stage is None:
            buff = None
            if self.nb_data > 0:
                buff = EsnliDataSet(split="TEST", nb_data=self.nb_data,
                                    cache_path=os.path.join(self.cache, "cleaned_data"))
            else:
                buff = EsnliDataSet(split="TEST", nb_data=-1,
                                    cache_path=os.path.join(self.cache, "cleaned_data"))
            self.test_set = buff

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
        batch = self.list2dict(batch)
        texts = self.t_add_sep(batch['premise'], batch['hypothesis'])
        input_ids, attention_mask = self.t_tokenize(texts)
        input_ids = self.t_tensor(input_ids)
        attention_mask = self.t_tensor(attention_mask)
        annotation = torch.stack(batch["annotation"], dim=0)
        labels = self.t_tensor(batch['label'])

        return {"input_ids": input_ids, "attention_masks": attention_mask, "labels": labels,
                "annotations": annotation}

    # maybe not usefull here
    def list2dict(self, batch):
        # convert list of dict to dict of list
        if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}
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

    # .cache folder >> the folder where everything will be saved
    cache = path.join(os.getcwd(), '.cache')

    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)

    # what model we should use the default is 1 >> the one created in this file
    parser.add_argument('-t', '--model_type', type=int, default=1)

    # default datadir >> ./.cache/dataset >> cache for our datamodule.
    parser.add_argument('-d', '--data_dir', default=path.join(cache, 'raw_data', 'e_snli'))

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

    # config for the regularization
    parser.add_argument('--reg_mul', type=float, default=0)  # the regularize terms
    parser.add_argument('--reg_lay', type=int, default=-1)  # the layer we want to regularize
    parser.add_argument('--lrate', type=float, default=5e-5)  # the learning rate for the training part

    args = parser.parse_args()

    if not args.exp:
        init_logging(color=False, cache_path=os.path.join(args.log_dir, args.experiment, args.version), oar_id="log_file_test")
    else:
        init_logging()

    # Summary information
    log.info(f'>>> Arguments: {json.dumps(vars(args), indent=4)}')

    # load the data for the training part
    dm = SNLIDataModule(
        cache=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nb_data=args.nb_data
    )
    dm.prepare_data()  # load the different data

    model = None
    if args.model_type == 1:
        model = BertNliRegu(criterion=nn.CrossEntropyLoss(),
                            reg_mul=args.reg_mul,
                            lr=args.lrate,
                            reg_lay=args.reg_lay,
                            exp=args.exp)

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
    early_stopping = cb.EarlyStopping(monitor="val_/loss", patience=5, verbose=args.exp,
                                      mode='min')
    model_checkpoint = cb.ModelCheckpoint(
        filename='best', monitor="val_/loss", mode='min',  # save the minimum val_loss
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
    log.info(f"performance of the model : {performance[0]}")
    log.info('Training finished')
