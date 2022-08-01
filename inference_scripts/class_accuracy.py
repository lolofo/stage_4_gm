# calculate the accuracy for each class
from os import path
from scipy.stats import ttest_ind
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


def p_value_signifiance(p):
    if p <= 0.001:
        return "* * *"
    elif p <= 0.01:
        return "* *"
    elif p <= 0.05:
        return "*"
    else :
        return "."


if __name__ == "__main__":
    init_logging()
    log.info("Calculus of the accuracy by class")
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

    count = {"entailement": 0, "neutral": 0, "contradiction": 0}
    count_preds = {"entailement": [], "neutral": [], "contradiction": []}
    LABELS = ["entailement", "neutral", "contradiction"]

    test_loader = dm.test_dataloader()
    spe_tok = torch.tensor([0, 101, 102]).to(DEVICE)
    it = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(dm.test_set) / args.batch_size)
        for batch in pbar:
            pbar.set_description("class accuracy")
            # loop for the inference on this dict
            ids = batch["input_ids"].to(DEVICE)
            annot = batch["annotations"].to(DEVICE)
            mk = batch["attention_masks"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=ids,
                            attention_mask=mk)

            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).type(torch.uint8)  # the prediction

            for b in range(args.batch_size):
                lb = LABELS[int(labels[b].item())]
                count[lb] += 1
                count_preds[lb].append(acc[b].item())

    log.info("the workforce of every class")
    log.info(count)

    log.info(f"the accuracy")
    acc = {}
    for k in LABELS:
        acc[k] = sum(count_preds[k])/count[k]
    log.debug(f"class accuracy : {acc}")

    log.info("signifiance test")
    test_table = [["Test", "T-stats", "p-value", "Signifiance"],
                  ["contradiction > entailement"],
                  ["contradictions > neutrals"]]

    t1 = ttest_ind(count_preds["contradiction"], count_preds["entailement"])
    t2 = ttest_ind(count_preds["contradiction"], count_preds["neutral"])

    test_table[1].append(t1[0])
    test_table[1].append(t1[1])
    test_table[1].append(p_value_signifiance(t1[1]))

    test_table[2].append(t2[0])
    test_table[2].append(t2[1])
    test_table[2].append(p_value_signifiance(t2[1]))

    dir = os.path.join(cache, "plots", f"class_accuracy")
    with open(os.path.join(dir, "class_accuracy.pickle"), "wb") as f:
        pickle.dump(test_table, f)




