# construction of a map for the attention on the token SEP in the CLS map
# objective, show that after the layer 3 there we start to have some attention on the layer SEP token

from os import path
import os
import sys

cwd = os.getcwd().split(os.path.sep)
while cwd[-1] != "stage_4_gm":
    os.chdir("..")
    cwd = os.getcwd().split(os.path.sep)
sys.path.extend([os.getcwd()])

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

from torch_set_up import DEVICE
from regularize_training_bert import SNLIDataModule
from regularize_training_bert import BertNliRegu
from training_bert import BertNliLight

if __name__ == "__main__":

    cwd = os.getcwd().split(os.path.sep)
    while cwd[-1] != "stage_4_gm":
        os.chdir("..")
        cwd = os.getcwd().split(os.path.sep)

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

    prem_att = []
    hyp_att = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            it += args.batch_size
            # loop for the inference on this dict
            ids = batch["input_ids"].to(DEVICE)
            annot = batch["annotations"].to(DEVICE)
            mk = batch["attention_masks"].to(DEVICE)
            output = model(input_ids=ids,
                           attention_mask=mk)["outputs"]

            attention_tensor = torch.stack(output.attentions, dim=1).sum(dim=1).sum(dim=1).to(DEVICE)
            cls_lines = attention_tensor[:, 0, :]  # [b , l, T]

            # normalize the cls_lines
            buff = cls_lines.clone()
            buff = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), buff, 1e30)
            mins = buff.min(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            maxs = cls_lines.max(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            cls_lines = (cls_lines - mins) / (maxs - mins)

            sep_pos = torch.isin(ids, torch.tensor([102]).to(DEVICE))

            # get the attention over the premise
            prem_pos = (torch.cumsum(sep_pos, dim=-1) == 0).type(torch.uint8)
            prem_att.append(torch.mul(prem_pos, cls_lines).sum(dim=-1) / (prem_pos.sum(dim=-1) - 1))

            # get the attention over the hypothesis
            hyp_pos = (torch.cumsum(sep_pos, dim=-1) == 1).type(torch.uint8)
            hyp_att.append(torch.mul(hyp_pos, cls_lines).sum(dim=-1) / (hyp_pos.sum(dim=-1) - 1))

    y = list(torch.concat(prem_att, dim=0).cpu().numpy()) + list(torch.concat(hyp_att, dim=0).cpu().numpy())
    print(f">> {len(list(torch.concat(prem_att, dim=0).cpu().numpy()))}")
    print(f">> {len(list(torch.concat(hyp_att, dim=0).cpu().numpy()))}")

    x = ["premise"] * int(len(y) / 2) + ["hypothesis"] * int(len(y) / 2)

    dir = os.path.join(os.getcwd(), ".cache", "plots", "cls_line_study")
    if not os.path.exists(dir):
        os.mkdir(dir)

    fig = plt.figure(figsize=(10, 10))
    ax = sns.boxplot(x=x, y=y)

    plt.savefig(os.path.join(dir, "attention_repartition.png"))
