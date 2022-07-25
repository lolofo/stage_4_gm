# preparation of the environment
from os import path
import os
import sys
cwd = os.getcwd().split(os.path.sep)
while cwd[-1] != "stage_4_gm":
    os.chdir("..")
    cwd = os.getcwd().split(os.path.sep)
sys.path.extend([os.getcwd()])

import argparse
import numpy as np
from tqdm import tqdm

from logger import log, init_logging
import torch
import pickle
import json

from torch_set_up import DEVICE

from regularize_training_bert import SNLIDataModule
from regularize_training_bert import BertNliRegu
from training_bert import BertNliLight


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


if __name__ == "__main__":

    cwd = os.getcwd().split(os.path.sep)
    while cwd[-1] != "stage_4_gm":
        os.chdir("..")
        cwd = os.getcwd().split(os.path.sep)
    print(f">> cwd >> {os.getcwd()}")



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
    # the log file
    init_logging(
        color=False,
        cache_path=os.path.join(cache, 'plots', f"reg_mul={args.reg_mul}"),
        oar_id=f"reg_mul_{args.reg_mul}_get_the_sep"
    )

    log.info(f'>>> Arguments: {json.dumps(vars(args), indent=4)}')

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
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # loop for the inference on this dict
            ids = batch["input_ids"].to(DEVICE)
            annot = batch["annotations"].to(DEVICE)
            mk = batch["attention_masks"].to(DEVICE)
            output = model(input_ids=ids,
                           attention_mask=mk)["outputs"]

            attention_tensor = torch.stack(output.attentions, dim=1).to(DEVICE)  # [b, l, h, T, T]
            if it == 0:
                log.info(f"look at some shapes for the first iteration of the batch")
                log.debug(f"attention_tensor.shape = {attention_tensor.shape}")

            attention_tensor = attention_tensor.sum(dim=1).sum(dim=1)  # sum over the layers and the head

            if it == 0:
                log.debug(f"attention_tensor.shape = {attention_tensor.shape}")

            sep_pos = torch.isin(ids, torch.tensor([102]).to(DEVICE)).to(DEVICE).type(torch.uint8)
            comb_pos = torch.isin(ids, torch.tensor([101, 102]).to(DEVICE)).to(DEVICE).type(torch.uint8)
            cls_pos = torch.isin(ids, torch.tensor([101]).to(DEVICE)).to(DEVICE).type(torch.uint8)

            if it == 0:
                log.debug(f"comb_pos.shape = {comb_pos.shape}")
                log.debug(f"sep_pos.shape = {sep_pos.shape}")

            sep_pos = sep_pos.unsqueeze(2).repeat(1, 1, 150)
            comb_pos = comb_pos.unsqueeze(2).repeat(1, 1, 150)
            cls_pos = cls_pos.unsqueeze(2).repeat(1, 1, 150)

            if it == 0:
                log.debug(f"sep_pos.shape (after unsqueeze) = {sep_pos.shape}")
                log.debug(f"comb_pos.shape (after unsqueeze) = {comb_pos.shape}")
                log.debug(f">>>>>>>>>>>>>{comb_pos}")

            sep_lines = torch.mul(attention_tensor, sep_pos).sum(dim=1)
            sep_lines = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), sep_lines, 0)
            comb_lines = torch.mul(attention_tensor, comb_pos).sum(dim=1)
            comb_lines = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), comb_lines, 0)
            cls_lines = torch.mul(attention_tensor, cls_pos).sum(dim=1)
            cls_lines = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), cls_lines, 0)


            sep_lines = sep_lines.clone()
            if it == 0:
                log.debug(f"sep_lines.shape = {sep_lines.shape}")

            buff = sep_lines.clone()
            buff = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), buff, 1e30)

            mins = buff.min(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            maxs = sep_lines.max(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            sep_lines = (sep_lines - mins) / (maxs - mins)
            sep_lines = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), sep_lines, 0)

            comb_lines = comb_lines.clone()
            if it == 0:
                log.debug(f"comb_lines.shape = {comb_lines.shape}")

            buff = comb_lines.clone()
            buff = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), buff, 1e30)

            mins = buff.min(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            maxs = comb_lines.max(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            comb_lines = (comb_lines - mins) / (maxs - mins)
            comb_lines = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), comb_lines, 0)

            cls_lines = cls_lines.clone()
            buff = cls_lines.clone()
            buff = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), buff, 1e30)

            mins = buff.min(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            maxs = cls_lines.max(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            cls_lines = (cls_lines - mins) / (maxs - mins)
            cls_lines = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), cls_lines, 0)

            y_hat.append(sep_lines.flatten())
            y_hat_combine.append(comb_lines.flatten())
            y_hat_cls.append(cls_lines.flatten())
            y.append(annot.flatten())
            IDS.append(ids.flatten())

            it += 1

        y = torch.concat(y, dim=0).cpu().numpy()
        y_hat = torch.concat(y_hat, dim=0).cpu().numpy()
        y_hat_combine = torch.concat(y_hat_combine, dim=0).cpu().numpy()
        y_hat_cls = torch.concat(y_hat_cls, dim=0).cpu().numpy()
        IDS = torch.concat(IDS, dim=0).cpu().numpy()

    idx = np.where(IDS == 0)[0]
    log.debug(f">> where ids==0 : {len(idx)}")
    y = np.delete(y, idx)
    y_hat = np.delete(y_hat, idx)
    y_hat_combine = np.delete(y_hat_combine, idx)
    y_hat_cls = np.delete(y_hat_cls, idx)

    # creation of the pickle
    log.info("pickle creation")
    d = {"y": y,
         "y_hat": y_hat}
    dir = os.path.join(cache, "plots", f"reg_mul={args.reg_mul}", "sep_map.pickle")
    with open(dir, "wb") as f:
        pickle.dump(d, f)

    d = {"y": y,
         "y_hat": y_hat_combine}
    dir = os.path.join(cache, "plots", f"reg_mul={args.reg_mul}", "comb_map.pickle")
    with open(dir, "wb") as f:
        pickle.dump(d, f)

    d = {"y": y,
         "y_hat": y_hat_cls}
    dir = os.path.join(cache, "plots", f"reg_mul={args.reg_mul}", "cls_map_baseline.pickle")
    with open(dir, "wb") as f:
        pickle.dump(d, f)

    log.info(">> DONE !")
