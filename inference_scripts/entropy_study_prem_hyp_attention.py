# distribution of the attention over the different lines


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
from torch_set_up import DEVICE
from regularize_training_bert import SNLIDataModule
from training_bert import BertNliLight

if __name__ == "__main__":
    init_logging()
    log.info("start entropy study : attention on the prem and the hyp")
    cache = path.join(os.getcwd(), '.cache')
    log.info(f"current directory {os.getcwd()}")

    parser = argparse.ArgumentParser()

    # .cache folder >> the folder where everything will be saved

    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-d', '--data_dir', default=path.join(cache, 'raw_data', 'e_snli'))
    parser.add_argument('-s', '--log_dir', default=path.join(cache, 'logs', 'igrida_trained'))
    parser.add_argument('-n', '--nb_data', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accelerator', type=str, default='auto')  # auto select GPU if exists

    args = parser.parse_args()

    ckp = path.join(args.log_dir, f"0", "best.ckpt")
    model = BertNliLight.load_from_checkpoint(ckp)
    model.to(DEVICE)
    model = model.eval()

    dm = SNLIDataModule(
        cache=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nb_data=args.nb_data
    )

    dm.setup(stage="test")

    # first column for the prem and the other for the hypothesis
    attention = {"entailement": np.zeros((12, 2, 3368)),
                 "neutral": np.zeros((12, 2, 3219)),
                 "contradiction": np.zeros((12, 2, 3237))
                 }

    entropy = {"entailement": np.zeros((12, 2, 3368)),
               "neutral": np.zeros((12, 2, 3219)),
               "contradiction": np.zeros((12, 2, 3237))
               }

    count = {"entailement": 0, "neutral": 0, "contradiction": 0}
    it = {"entailement": 0, "neutral": 0, "contradiction": 0}

    LABELS = ["entailement", "neutral", "contradiction"]
    INF = 1e30

    test_loader = dm.test_dataloader()
    spe_tok = torch.tensor([0, 101, 102]).to(DEVICE)
    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(dm.test_set) / args.batch_size)
        for batch in pbar:
            pbar.set_description("get the attention on the prem and the hyp")
            # loop for the inference on this dict
            ids = batch["input_ids"].to(DEVICE)
            annot = batch["annotations"].to(DEVICE)
            mk = batch["attention_masks"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            output = model(input_ids=ids,
                           attention_mask=mk)["outputs"]

            # the specials tokens
            special_tokens = [0, 101, 102]
            target = torch.tensor(special_tokens).to(DEVICE)
            spe_tok_mask = torch.isin(ids, target).type(torch.uint8)
            spe_tok_mask = spe_tok_mask[:, None, :].repeat(1, 12, 1)

            # get the positions of the specials tokens
            buff = spe_tok_mask.cumsum(dim=-1)
            prem_position = (buff == 1).type(torch.uint8)
            hyp_position = (buff == 2).type(torch.uint8)

            # process the attention_tensor
            attention_tensor = torch.stack(output.attentions, dim=1)  # shape [b, l, h, T, T]
            pad = torch.tensor([0]).to(DEVICE)
            pad_mask = torch.logical_not(torch.isin(ids, pad)).type(torch.uint8).unsqueeze(1).unsqueeze(1).unsqueeze(
                1).repeat(1, 12, 12, 150, 1)
            pad_mask = torch.transpose(pad_mask, dim0=3, dim1=4)
            attention_tensor = torch.mul(attention_tensor, pad_mask)

            # construction of the weights
            a_hat = attention_tensor[:, :, :, :, :].sum(dim=2) / 12  # mean over the heads
            a_hat = a_hat.sum(dim=2)  # sum over the lines.
            a_hat = torch.softmax(a_hat - INF * spe_tok_mask, dim=-1)

            a_hat_prem = torch.mul(a_hat, prem_position).sum(dim=-1)
            a_hat_hyp = torch.mul(a_hat, hyp_position).sum(dim=-1)

            for b in range(args.batch_size):
                lb = LABELS[labels[b]]
                count[lb] += 1
                pos = it[lb]
                attention[lb][:, 0, pos] += a_hat_prem[b, :].cpu().numpy()
                attention[lb][:, 1, pos] += a_hat_hyp[b, :].cpu().numpy()
                it[lb] += 1

    log.info("save the different dictionnaries")
    dir = os.path.join(cache, "plots", f"entropy_study")

    with open(os.path.join(dir, "attention_prem_hyp.pickle"), "wb") as f:
        pickle.dump(attention, f)
