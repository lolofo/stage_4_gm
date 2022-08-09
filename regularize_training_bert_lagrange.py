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
from torchmetrics import AveragePrecision

from transformers import BertModel
from transformers import BertTokenizer

import os
from os import path

from pytorch_lightning import callbacks as cb

from modules import transforms as t
from logger import log, init_logging

tk = BertTokenizer.from_pretrained('bert-base-uncased')

#############
### model ###
#############

# constants for numerical stabilities
EPS = 1e-16
INF = 1e30


def L2D(batch):
    # convert list of dict to dict of list
    if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}
    return {k: [row[k] for row in batch] for k in batch[0]}


class BertNliRegu(pl.LightningModule):

    def __init__(self, freeze_bert=False,
                 criterion=nn.CrossEntropyLoss(),
                 reg_mul=0,
                 pen_type: str = "lasso",
                 lr=5e-5,
                 exp: bool = False):

        super().__init__()
        self.exp = exp
        self.save_hyperparameters("reg_mul", ignore=["exp"])

        self.lr = lr

        # bert layer
        # the bert layer will return the layer will return the attention weights
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True
                                              # return the attention weights
                                              )

        # classifier head
        self.classifier = nn.Sequential(  # fully connected layer
            nn.Linear(in_features=768, out_features=3)
        )

        # multiplier of the reg term
        # if this term is high >> high regularization
        self.reg_mul = reg_mul
        self.pen_type = pen_type
        self.criterion = criterion

        # metrics
        self.acc = nn.ModuleDict({
            'TRAIN': Accuracy(num_classes=3),
            'VAL': Accuracy(num_classes=3),
            'TEST': Accuracy(num_classes=3)
        })

        self.auc = nn.ModuleDict({
            'TRAIN': AUROC(pos_label=1, average="micro"),
            'VAL': AUROC(pos_label=1, average="micro"),
            'TEST': AUROC(pos_label=1, average="micro")
        })

        self.auprc = nn.ModuleDict({
            'TRAIN': AveragePrecision(pos_label=1, average="micro"),
            'VAL': AveragePrecision(pos_label=1, average="micro"),
            'TEST': AveragePrecision(pos_label=1, average="micro")
        })

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

    ###############################
    ### regularization function ###
    ###############################

    def entropy_regu(self, outputs, input_ids, h_annot, pen_type: str = "lasso"):
        # the mask for the specials tokens
        spe_ids = torch.tensor([0, 101, 102]).to(self.device)
        spe_tok_mask = torch.isin(input_ids, spe_ids)

        # process the attention_tensor
        attention_tensor = torch.stack(outputs.attentions, dim=1)  # shape [b, l, h, T, T]
        pad = torch.tensor([0]).to(self.device)
        pad_mask = torch.logical_not(torch.isin(input_ids, pad)).type(torch.uint8) \
            .unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 12, 12, 150, 1)
        pad_mask = torch.transpose(pad_mask, dim0=3, dim1=4)
        attention_tensor = torch.mul(attention_tensor, pad_mask)

        # the entropia calculus
        a_hat = attention_tensor[:, 3:10, :, :, :]  # select layers from 4 to 10
        a_hat = a_hat.sum(dim=2) / 12  # mean head agregation
        a_hat = a_hat.sum(dim=1)  # sum over the layers
        a_hat = a_hat.sum(dim=1)  # line agregation
        a_hat_4_10 = torch.softmax(a_hat - INF * spe_tok_mask, dim=-1)
        ent_4_10 = (-a_hat_4_10 * torch.log(a_hat_4_10 + EPS)).sum(dim=-1)
        nb_tokens = torch.logical_not(spe_tok_mask).type(torch.float).sum(dim=-1)
        log_t = torch.log(nb_tokens)
        h = ent_4_10 / log_t

        if pen_type == "lasso":
            pen = (torch.abs(h - h_annot)).mean(dim=0)
        else:
            pen = (torch.square(h - h_annot)).mean(dim=0)

        # return the penalisation score and the model annotations
        if pen >= 50:
            log.debug(f"h (entropy calculated) : {h}")

        return {"pen": pen, "scores": a_hat_4_10}

    #######################
    ### steps functions ###
    #######################

    def step(self, batch, batch_idx):
        # the batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_masks"]
        labels = batch["labels"]
        # forward of the model
        buff = self.forward(input_ids, attention_mask)
        logits = buff["logits"]
        outputs = buff["outputs"]
        # the loss
        loss = self.criterion(logits, labels)
        reg_term = self.entropy_regu(outputs=outputs,
                                     input_ids=input_ids,
                                     h_annot=batch["H_annot"],
                                     pen_type=self.pen_type)
        loss += self.reg_mul * reg_term["pen"]

        # the probabilities
        class_pred = torch.softmax(logits, dim=1)

        # score for the metrics
        padding = torch.tensor([0]).to(self.device).clone().detach()
        non_pad_pos = torch.logical_not(torch.isin(torch.flatten(input_ids), padding)).clone().detach()
        a_hat = torch.flatten(reg_term["scores"])[non_pad_pos].clone().detach()
        a_true = torch.flatten(batch["annotations"])[non_pad_pos].clone().detach()

        return {'loss': loss, 'preds': class_pred, 'target': labels, 'reg_term': reg_term["pen"],
                'auc': (a_hat, a_true)}

    def step_end(self, output, stage: str):
        step_acc = self.acc[stage](output['preds'], output['target'])
        step_auc = self.auc[stage](output['auc'][0], output['auc'][1])
        step_auprc = self.auprc[stage](output['auc'][0], output['auc'][1])
        if stage == "VAL":
            # for the EarlyStopping
            epoch_bool = True
        else:
            epoch_bool = False
        self.log(f"{stage}_/loss", output['loss'], on_step=True, on_epoch=epoch_bool, logger=True)
        self.log(f"{stage}_/acc", step_acc, on_step=True, on_epoch=epoch_bool, logger=True, prog_bar=True)
        self.log(f"{stage}_/reg", output["reg_term"], on_step=True, on_epoch=epoch_bool, logger=True, prog_bar=True)
        self.log(f"{stage}_/auc", step_auc, on_step=True, on_epoch=epoch_bool, logger=True)
        self.log(f"{stage}_/auprc", step_auprc, on_step=True, on_epoch=epoch_bool, logger=True)

    def end_epoch(self, stage):
        d = dict()
        d[f"{stage}_acc"] = round(self.acc[stage].compute().item(), 4)
        d[f"{stage}_auc"] = round(self.auc[stage].compute().item(), 4)
        d[f"{stage}_auprc"] = round(self.auprc[stage].compute().item(), 4)
        log.info(f"Epoch : {self.current_epoch} >> {stage}_metrics >> {d}")

    ####################
    ### the training ###
    ####################
    def on_train_start(self):
        # init the values for the matrix board
        init_hp_metrics = {'hp_/acc': 0, 'hp_/auc': 0, 'hp_/reg': 0, 'hp_/loss': 0, 'hp_/auprc': 0}
        self.logger.log_hyperparams(self.hparams, init_hp_metrics)

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx)

    def training_step_end(self, output):
        self.step_end(output=output, stage="TRAIN")

    def on_train_epoch_end(self):
        return self.end_epoch(stage="TRAIN")

    ######################
    ### the validation ###
    ######################
    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, batch_idx)

    def validation_step_end(self, output):
        self.step_end(output=output, stage="VAL")

    def on_validation_epoch_end(self):
        return self.end_epoch(stage="VAL")

    ################
    ### the test ###
    ################
    def test_step(self, test_batch, batch_idx):
        return self.step(test_batch, batch_idx)

    def test_step_end(self, output):
        test_acc = self.acc["TEST"](output['preds'], output['target'])
        test_auc = self.auc["TEST"](output["auc"][0], output["auc"][1])
        test_auprc = self.auprc["TEST"](output["auc"][0], output["auc"][1])

        self.log("hp_/reg", output["reg_term"], on_step=False, on_epoch=True, logger=True)
        self.log("hp_/loss", output["loss"], on_step=False, on_epoch=True, logger=True)
        self.log("hp_/acc", test_acc, on_step=False, on_epoch=True, logger=True)
        self.log("hp_/auc", test_auc, on_step=False, on_epoch=True, logger=True)
        self.log("hp_/auprc", test_auprc, on_step=False, on_epoch=True, logger=True)


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
        if stage == 'fit' or stage is None:
            buff = None
            if self.nb_data > 0:
                buff = EsnliDataSet(split="TRAIN", nb_data=self.nb_data,
                                    cache_path=os.path.join(self.cache, "cleaned_data"))
            else:
                buff = EsnliDataSet(split="TRAIN", nb_data=-1, cache_path=os.path.join(self.cache, "cleaned_data"))
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
                buff = EsnliDataSet(split="TEST", nb_data=-1, cache_path=os.path.join(self.cache, "cleaned_data"))
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

        # don't put the punctuation into the annotation
        punct_ids = torch.tensor(list(range(999, 1037)))
        punct_pos = torch.logical_not(torch.isin(input_ids, punct_ids)).type(torch.uint8)
        annotation = torch.mul(annotation, punct_pos)

        # calculation of the entropy of the annotation
        spe_ids = torch.tensor([0, 101, 102])
        spe_tok_mask = torch.isin(input_ids, spe_ids)

        # renormalize the annotation
        a_s = annotation.sum(dim=-1).type(torch.float)
        t_s = torch.logical_not(spe_tok_mask).type(torch.float).sum(dim=-1)
        h_annot = torch.log(a_s) / torch.log(t_s)

        if (h_annot > 100).type(torch.uint8).sum() >= 1:
            log.debug(f"batch : {batch}")
            log.debug(f"a_s : {a_s}")
            log.debug(f"t_s : {t_s}")

        return {
            "input_ids": input_ids,
            "attention_masks": attention_mask,
            "labels": labels,
            "annotations": annotation,
            "H_annot": h_annot
        }

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
    parser.add_argument('--exp', action='store_true')

    # save in [args.log_dir]/[args.experiments]/[args.version] allow good tensorboard
    parser.add_argument('--experiment', type=str, default='test')
    parser.add_argument('--version', type=str, default='0.0')

    # config for cluster distribution
    parser.add_argument('--num_workers', type=int,
                        default=get_num_workers())  # auto select appropriate cores in machine
    parser.add_argument('--accelerator', type=str, default='auto')  # auto select GPU if exists

    # config for the regularization
    parser.add_argument('--reg_mul', type=float, default=0)  # the regularize terms
    parser.add_argument('--pen_type', type=str, default="lasso")  # how to regularize lasso or mse
    parser.add_argument('--lrate', type=float, default=5e-5)  # the learning rate for the training part

    args = parser.parse_args()

    if args.exp:
        init_logging(color=not args.exp, cache_path=os.path.join(args.log_dir, args.experiment, args.version),
                     oar_id=f"RegMul={args.reg_mul}")
    else:
        init_logging()

    # Summary information
    log.info(f'>>> Arguments: {json.dumps(vars(args), indent=4)}')

    # load the data for the training part
    dm = SNLIDataModule(cache=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
                        nb_data=args.nb_data)
    dm.prepare_data()  # load the different data

    model = None
    if args.model_type == 1:
        model = BertNliRegu(criterion=nn.CrossEntropyLoss(),
                            reg_mul=args.reg_mul,
                            lr=args.lrate,
                            pen_type=args.pen_type,
                            exp=args.exp)

    ######################
    ### trainer config ###
    ######################

    # set the direction to visualize the logs of the training
    # the visualization will be done with tensorboard.
    logger = TensorBoardLogger(save_dir=args.log_dir,  # the main log folder
                               name=args.experiment,  # name of the log >> related to the name of the model we use
                               version=args.version,  # version of the log
                               default_hp_metric=False  # deactivate hp_metric on tensorboard visualization
                               )
    # logger = TensorBoardLogger(name=args.log_dir, save_dir=log_dir + '/')

    # call back
    early_stopping = cb.EarlyStopping(monitor="VAL_/loss", patience=5, verbose=args.exp, mode='min')
    model_checkpoint = cb.ModelCheckpoint(filename='best',
                                          monitor="VAL_/loss",
                                          mode='min',  # save the minimum val_loss
                                          )

    trainer = pl.Trainer(max_epochs=args.epoch,
                         accelerator=args.accelerator,  # auto use gpu
                         enable_progress_bar=not args.exp,  # hide progress bar in experimentation
                         log_every_n_steps=1,
                         default_root_dir=args.log_dir,
                         logger=logger,
                         callbacks=[early_stopping, model_checkpoint],
                         detect_anomaly=not args.exp)

    #############################
    ### training of the model ###
    #############################
    dm.setup(stage='fit')
    trainer.fit(model, datamodule=dm)

    dm.setup(stage='test')
    performance = trainer.test(ckpt_path='best', datamodule=dm)
    log.info(f"performance of the model : {performance[0]}")
    log.info('Training finished')
