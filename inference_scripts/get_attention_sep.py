# construction of a map for the attention on the token SEP in the CLS map
# objective, show that after the layer 3 there we start to have some attention on the layer SEP token

from os import path
import os

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import seaborn as sns
from tqdm import tqdm

import torch
import pickle
import json

from torch_set_up import DEVICE
from regularize_training_bert import SNLIDataModule
from regularize_training_bert import BertNliRegu
from training_bert import BertNliLight

if __name__ == "__main__":
    cwd = os.getcwd().split(os.path.sep)
    while cwd[-1] != "stage_4_gm":
        os.chdir("..")
        cwd = os.getcwd().split(os.path.sep)
    print(f">> cwd >> {os.getcwd()}")

    sns.set_theme()
    parser = argparse.ArgumentParser()

    # .cache folder >> the folder where everything will be saved
    cache = path.join(os.getcwd(), '.cache')

    parser.add_argument('-b', '--batch_size', type=int, default=4)

    # default datadir >> ./.cache/dataset >> cache for our datamodule.
    parser.add_argument('-d', '--data_dir', default=path.join(cache, 'raw_data', 'e_snli'))

    # log_dir for the logger
    parser.add_argument('-s', '--log_dir', default=path.join(cache, 'logs', 'igrida_trained'))

    parser.add_argument('-n', '--nb_data', type=int, default=-1)

    # config for cluster distribution
    parser.add_argument('--num_workers', type=int,
                        default=4)  # auto select appropriate cores in machine
    parser.add_argument('--accelerator', type=str, default='auto')  # auto select GPU if exists

    # config for the regularization
    parser.add_argument('--reg_mul', type=float, default=0)  # the regularize terms
    parser.add_argument('--reg_lay', type=int, default=-1)  # the layer we want to regularize
    parser.add_argument('--lrate', type=float, default=5e-5)  # the learning rate for the training part

    args = parser.parse_args()

    # load the model from the right checkpoint
    model = None
    if args.reg_mul == 0:
        ckp = path.join(args.log_dir, f"0", "best.ckpt")
        model = BertNliLight.load_from_checkpoint(ckp)
    else:
        ckp = path.join(args.log_dir, f"reg_mul={args.reg_mul}", "best.ckpt")
        model = BertNliRegu.load_from_checkpoint(ckp)

    model.to(DEVICE)
    model = model.eval()

    dm = SNLIDataModule(
        cache=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nb_data=args.nb_data
    )

    dm.setup(stage="test")

    IDS = []
    y = []
    y_hat = []
    y_hat_combine = []
    y_hat_cls = []

    d = {}
    test_loader = dm.test_dataloader()
    spe_tok = torch.tensor([0, 101, 102]).to(DEVICE)
    it = 0

    buff = torch.zeros((12, 1)).to(DEVICE)
    cls_buff = torch.zeros((12, 1)).to(DEVICE)

    with torch.no_grad():
        for batch in tqdm(test_loader):
            it += args.batch_size
            # loop for the inference on this dict
            ids = batch["input_ids"].to(DEVICE)
            annot = batch["annotations"].to(DEVICE)
            mk = batch["attention_masks"].to(DEVICE)
            output = model(input_ids=ids,
                           attention_mask=mk)["outputs"]

            attention_tensor = torch.stack(output.attentions, dim=1).sum(dim=2).to(DEVICE) / 12  # sum over the heads
            cls_lines = attention_tensor[:, :, 0, :]  # [b , l, T]
            sep_pos = torch.isin(ids, torch.tensor([102]).to(DEVICE)).to(DEVICE).type(torch.uint8)  # [b, T]
            sep_pos = sep_pos[:, None, :]
            sep_pos = sep_pos.repeat(1, 12, 1)  # repeat over the layers
            sep_attention = torch.mul(sep_pos, cls_lines).sum(dim=0).sum(dim=-1)

            cls_attention = attention_tensor[:, :, 0, 0]  # attention on the CLS token for the CLS line
            cls_attention = cls_attention.sum(dim=0)

            buff[:, 0] += sep_attention
            cls_buff[:, 0] += cls_attention

    buff = buff.cpu().numpy()
    cls_buff = cls_buff.cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    plt.plot(list(range(1, 13)), buff[:, 0] / it, label="SEP -- at")
    plt.plot(list(range(1, 13)), cls_buff[:, 0] / it, label="CLS -- at")
    plt.hlines(1, 1, 12, linestyles="dashed", colors="r")
    plt.xlabel("layer")
    plt.ylabel("attention")
    plt.legend()

    dir = os.path.join(os.getcwd(), ".cache", "plots", "cls_line_study")
    if not os.path.exists(dir):
        os.mkdir(dir)

    plt.savefig(os.path.join(dir, "sep_cls_attention_over_layers"))
