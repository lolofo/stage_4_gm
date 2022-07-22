# preparation of the environment
from os import path
import os
import argparse
import numpy as np

from tqdm import tqdm

from modules.logger import log, init_logging
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

    init_logging(
        color=False,
        cache_path=os.path.join(cache, 'plots', f"reg_mul={args.reg_mul}"),
        oar_id=f"reg_mul_{args.reg_mul}"
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
    y_hat_bis = []

    d = {}
    test_loader = dm.test_dataloader()
    spe_tok = torch.tensor([0, 101, 102]).to(DEVICE)

    with torch.no_grad():
        for batch in tqdm(test_loader):
            # loop for the inference on this dict
            ids = batch["input_ids"].to(DEVICE)
            annot = batch["annotations"].to(DEVICE)
            mk = batch["attention_masks"].to(DEVICE)
            output = model(input_ids=ids,
                           attention_mask=mk)["outputs"]

            attention_tensor = torch.stack(output.attentions, dim=1)
            # sum over the lines and the heads
            # no selection here
            cls_lines = attention_tensor[:, :, :, 0, :].sum(dim=1).sum(dim=1)

            # replace the specials tokens by zero
            cls_lines = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), cls_lines, 0)

            # when the min is calculated --> don't take into account the specials tokens
            buff = cls_lines.clone()
            buff = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), buff, 1e30)

            mins = buff.min(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            maxs = cls_lines.max(dim=-1)[0].unsqueeze(1).repeat(1, 150)

            cls_lines = (cls_lines - mins) / (maxs - mins)

            # get back to zeros the specials tokens
            cls_lines = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), cls_lines, 0)

            sum_agreg = attention_tensor[:, :, :, :, :].sum(dim=1).sum(dim=1).sum(dim=1)
            # replace the specials tokens by zero
            sum_agreg = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), sum_agreg, 0)

            # when the min is calculated --> don't take into account the specials tokens
            buff = sum_agreg.clone()
            buff = torch.where(torch.logical_not(torch.isin(ids, spe_tok)), buff, 1e30)

            mins = buff.min(dim=-1)[0].unsqueeze(1).repeat(1, 150)
            maxs = sum_agreg.max(dim=-1)[0].unsqueeze(1).repeat(1, 150)

            sum_agreg = (sum_agreg - mins) / (maxs - mins)

            y_hat.append(cls_lines.flatten())
            y_hat_bis.append(sum_agreg.flatten())
            y.append(annot.flatten())
            IDS.append(ids.flatten())

        y = torch.concat(y, dim=0).cpu().numpy()
        y_hat = torch.concat(y_hat, dim=0).cpu().numpy()
        y_hat_bis = torch.concat(y_hat_bis, dim=0).cpu().numpy()
        IDS = torch.concat(IDS, dim=0).cpu().numpy()

    log.debug(f">> len(y) : {len(y)}")
    log.debug(f">> len(y_hat) : {len(y_hat)}")
    log.debug(f">> len(y_hat_bis) : {len(y_hat_bis)}")
    log.debug(f">> IDS : {len(IDS)}")
    log.debug(f">> ckeck dim : {len(y) == len(y_hat)}")

    idx = np.where(IDS == 0)[0]
    log.debug(f">> where ids==0 : {len(idx)}")
    y = np.delete(y, idx)
    y_hat = np.delete(y_hat, idx)
    y_hat_bis = np.delete(y_hat_bis, idx)

    log.info(">> after selection")
    log.debug(f">> len(y) : {len(y)}")
    log.debug(f">> len(y_hat) : {len(y_hat)}")
    log.debug(f">> len(y_hat_bis) : {len(y_hat_bis)}")

    # creation of the pickle
    log.info("pickle creation")
    d = {"y": y,
         "y_hat": y_hat}

    dir = os.path.join(cache, "plots", f"reg_mul={args.reg_mul}", "cls_map.pickle")

    with open(dir, "wb") as f:
        pickle.dump(d, f)

    d = {"y": y,
         "y_hat": y_hat_bis}

    dir = os.path.join(cache, "plots", f"reg_mul={args.reg_mul}", "sum_agreg_map.pickle")

    with open(dir, "wb") as f:
        pickle.dump(d, f)

    log.info(">> DONE !")
