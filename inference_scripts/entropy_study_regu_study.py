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
from regularize_training_bert import BertNliRegu
from regularize_training_bert_ent_modify import BertNliRegu as BertNliReguModif

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
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--modif',  action='store_true')

    args = parser.parse_args()

    log.info(f"start entropy study reg_study : REG MUL = {args.reg_mul}")
    log.info(f"modif : {args.modif}")

    if not args.modif:
        ckp = path.join(args.log_dir, "regu_study", "layer_4_10", f"mul={args.reg_mul}", "checkpoints", "best.ckpt")
        model = BertNliRegu.load_from_checkpoint(ckp)
        model.to(DEVICE)
        model = model.eval()
    else:
        ckp = path.join(args.log_dir, "regu_study", "layer_4_10", f"modif_mul={args.reg_mul}", "checkpoints", "best.ckpt")
        model = BertNliReguModif.load_from_checkpoint(ckp)
        model.to(DEVICE)
        model = model.eval()



    dm = SNLIDataModule(
        cache=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nb_data=args.nb_data
    )

    dm.setup(stage="test")

    a_true = {"entailement": [], "neutral": [], "contradiction": []
              }

    layers_4_10 = {"entailement": [], "neutral": [], "contradiction": [],
                   "entropy": {"entailement": [0], "neutral": [0], "contradiction": [0]},
                   "modif_entropy": {"entailement": [0], "neutral": [0], "contradiction": [0]}
                   }

    IDS = {"entailement": [], "neutral": [], "contradiction": []}
    count = {"entailement": 0, "neutral": 0, "contradiction": 0}

    LABELS = ["entailement", "neutral", "contradiction"]
    INF = 1e30

    test_loader = dm.test_dataloader()
    spe_tok = torch.tensor([0, 101, 102]).to(DEVICE)
    it = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(dm.test_set) / args.batch_size)
        for batch in pbar:
            pbar.set_description("calculation of the attentions_weight")
            # loop for the inference on this dict
            ids = batch["input_ids"].to(DEVICE)
            annot = batch["annotations"].to(DEVICE)
            mk = batch["attention_masks"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # don't put the punctuation in our model.
            punct_ids = torch.tensor(list(range(999, 1037))).to(DEVICE)
            punct_pos = torch.logical_not(torch.isin(ids, punct_ids)).type(torch.uint8)
            annot = torch.mul(annot, punct_pos)

            output = model(input_ids=ids,
                           attention_mask=mk)["outputs"]

            # the specials tokens
            special_tokens = [0, 101, 102]
            target = torch.tensor(special_tokens).to(DEVICE)
            spe_tok_mask = torch.isin(ids, target).type(torch.uint8)

            # process the attention_tensor
            attention_tensor = torch.stack(output.attentions, dim=1)  # shape [b, l, h, T, T]

            pad = torch.tensor([0]).to(DEVICE)
            pad_mask = torch.logical_not(torch.isin(ids, pad)).type(torch.uint8).unsqueeze(1).unsqueeze(1).unsqueeze(
                1).repeat(1, 12, 12, 150, 1)
            pad_mask = torch.transpose(pad_mask, dim0=3, dim1=4)
            attention_tensor = torch.mul(attention_tensor, pad_mask)

            # layer 4 to 10
            a_hat = attention_tensor[:, 3:10, :, :, :].clone()  # select only some layers
            a_hat = a_hat.sum(dim=2) / 12  # mean head agregation
            a_hat = a_hat.sum(dim=1)
            a_hat = a_hat.sum(dim=1)  # line agregation
            a_hat_4_10 = torch.softmax(a_hat - INF * spe_tok_mask, dim=-1)
            ent_4_10 = (-a_hat_4_10 * torch.log(a_hat_4_10 + 1e-16)).sum(dim=-1)

            nb_tokens = torch.logical_not(spe_tok_mask).type(torch.float).sum(dim=-1)
            log_t = torch.log(nb_tokens)
            h = ent_4_10 / log_t
            ent_modif = (0.25 - torch.mul(h, 1.0 - h))

            for b in range(args.batch_size):
                lb = LABELS[int(labels[b].item())]
                IDS[lb].append(ids[b, :].cpu())
                count[lb] += 1

                # add the weights
                a_true[lb].append(annot[b, :].cpu())
                layers_4_10[lb].append(a_hat_4_10[b, :].cpu())

                # add the entropy
                layers_4_10["entropy"][lb][0] += h[b].item()
                layers_4_10["modif_entropy"][lb][0] += ent_modif[b].item()

            it += 1

    # concat the different tensors to proceed a macro approach
    pbar = tqdm(LABELS, total=len(LABELS))
    for k in pbar:
        pbar.set_description(f"now lets get a good format")
        IDS[k] = torch.concat(IDS[k], dim=0).cpu().numpy()
        idx = np.where(IDS[k] == 0)[0]

        a_true[k] = torch.concat(a_true[k], dim=0).cpu().numpy()
        a_true[k] = np.delete(a_true[k], idx)

        layers_4_10[k] = torch.concat(layers_4_10[k], dim=0).cpu().numpy()
        layers_4_10[k] = np.delete(layers_4_10[k], idx)
        layers_4_10["entropy"][k][0] /= count[k]
        layers_4_10["modif_entropy"][k][0] /= count[k]

    # proceed some statistics
    log.info("some statistics")
    for k in LABELS:
        a = []
        a.append(len(a_true[k]))
        a.append(len(layers_4_10[k]))
        log.debug(f">> {k} >> {a} ")

    # prepare the environment
    dir = os.path.join(cache, "plots", f"regu_study", "layer_4_10")
    if args.modif :
        if not os.path.exists(os.path.join(dir, f"modif_mul={args.reg_mul}")):
            os.mkdir(os.path.join(dir, f"modif_mul={args.reg_mul}"))
        dir = os.path.join(dir, f"modif_mul={args.reg_mul}")
    else :
        if not os.path.exists(os.path.join(dir, f"mul={args.reg_mul}")):
            os.mkdir(os.path.join(dir, f"mul={args.reg_mul}"))
        dir = os.path.join(dir, f"mul={args.reg_mul}")

    with open(os.path.join(dir, "a_true_head_mean.pickle"), "wb") as f:
        pickle.dump(a_true, f)

    with open(os.path.join(dir, "layers_4_10_head_mean.pickle"), "wb") as f:
        pickle.dump(layers_4_10, f)
