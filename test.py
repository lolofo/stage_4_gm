INF = 1e30

import os
from os import path


# set the repository to the git repository
cwd = os.getcwd().split(os.path.sep)
while cwd[-1] != "stage_4_gm":
    os.chdir("..")
    cwd = os.getcwd().split(os.path.sep)
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from torch_set_up import DEVICE
from tqdm import tqdm
from training_bert import BertNliLight
from regularize_training_bert_lagrange import SNLIDataModule
from logger import log, init_logging


if __name__ == "__main__":
    dm = SNLIDataModule(cache=os.path.join(".cache", "raw_data", "e_snli"), batch_size=4, num_workers=4,
                        nb_data=-1)

    dm.setup(stage="fit")
    train_dl = dm.train_dataloader()
    for batch in tqdm(train_dl):
        buff = batch
        if (batch["H_annot"] >= 50).type(torch.uint8).sum() >= 1:
            log.debug("problem")
    val_dl = dm.val_dataloader()
    for batch in tqdm(val_dl):
        buff = batch
        if (batch["H_annot"] >= 50).type(torch.uint8).sum() >= 1:
            log.debug("problem")

    dm.setup(stage="test")
    test_dl = dm.test_dataloader()
    for batch in tqdm(test_dl):
        buff = batch
        if (batch["H_annot"] >= 50).type(torch.uint8).sum() >= 1:
            log.debug("problem")