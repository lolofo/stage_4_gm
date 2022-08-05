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
from regularize_training_bert import SNLIDataModule, BertNliRegu

if __name__ == "__main__":
    init_logging()
    cache = path.join(os.getcwd(), '.cache')

    parser = argparse.ArgumentParser()

    # .cache folder >> the folder where everything will be saved

    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-d', '--data_dir', default=path.join(cache, 'raw_data', 'e_snli'))
    parser.add_argument('-s', '--log_dir', default=path.join(cache, 'logs', 'igrida_trained'))
    parser.add_argument('--reg_mul', type=float, default=0.0)
    parser.add_argument('-n', '--nb_data', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accelerator', type=str, default='auto')  # auto select GPU if exists

    args = parser.parse_args()

    log.info(f"start entropy study reg_study : REG MUL = {args.reg_mul}")

    ckp = path.join(args.log_dir, "regu_study", "layer_4_10", f"mul={args.reg_mul}", "checkpoints", "best.ckpt")
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

    # first column for the prem and the other for the hypothesis
    cosine_distribution = {
        "entailement": np.zeros((13, 1)),
        "neutral": np.zeros((13, 1)),
        "contradiction": np.zeros((13, 1))
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
            pbar.set_description("get the cosines")
            # loop for the inference on this dict
            ids = batch["input_ids"].to(DEVICE)
            annot = batch["annotations"].to(DEVICE)
            mk = batch["attention_masks"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            nb_tokens = mk.sum(dim=-1)

            output = model(input_ids=ids,
                           attention_mask=mk)["outputs"]

            hidden_states = torch.stack(output.hidden_states, dim=1)

            # compute the norm for the den
            hd_states_norms = torch.norm(hidden_states, p=2, dim=-1)
            output_1 = hd_states_norms.unsqueeze(-1)
            output_2 = hd_states_norms.unsqueeze(2)
            norm_den = torch.matmul(output_1, output_2)

            # compute the dot product between all the hidden states
            output_1 = hidden_states.clone()
            output_2 = torch.transpose(hidden_states.clone(), dim0=2, dim1=3)
            scalar_dot = torch.matmul(output_1, output_2)

            cos_tensor = scalar_dot / norm_den

            for b in range(args.batch_size):
                lb = LABELS[labels[b]]
                m = nb_tokens[b].item()
                buff = torch.flatten(cos_tensor[b, :, 0:m, 0:m], start_dim=1, end_dim=2).cpu().numpy()
                cosine_distribution[lb] = np.concatenate((cosine_distribution[lb], buff), axis=-1)

        log.info("just to check")
        log.debug(f"{cosine_distribution['entailement'].shape}")

        dir = os.path.join(cache, "plots", f"regu_study", "layer_4_10")
        if not os.path.exists(os.path.join(dir, f"mul={args.reg_mul}")):
            os.mkdir(os.path.join(dir, f"mul={args.reg_mul}"))
        dir = os.path.join(dir, f"mul={args.reg_mul}")

        with open(os.path.join(dir, "cosine_distribution.pickle"), "wb") as f:
            pickle.dump(cosine_distribution, f)