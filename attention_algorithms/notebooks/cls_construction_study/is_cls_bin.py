import torch
from tqdm import tqdm

from training_bert import BertNliLight
from custom_data_set import SnliDataset
from custom_data_set import test_dir, dev_dir
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from os import path

import os
from os import path
import matplotlib.pyplot as plt
from attention_algorithms.attention_metrics import default_plot_colormap
from torch_set_up import DEVICE


# set the repository to the git repository
def main_prepare():
    cwd = os.getcwd().split(os.path.sep)
    while cwd[-1] != "stage_4_gm":
        os.chdir("..")
        cwd = os.getcwd().split(os.path.sep)

    # the folder where we will save our data
    plots_folder = os.path.join(os.getcwd(), '.cache', 'plots')
    graph_folder = path.join(plots_folder, "gradient_cls")
    if not path.exists(path.join(plots_folder, "gradient_cls")):
        os.mkdir(path.join(plots_folder, "gradient_cls"))

    return os.getcwd(), graph_folder


if __name__ == "__main__":

    ckp = path.join(".cache", "logs", "igrida_trained", "0", "best.ckpt")
    model = BertNliLight.load_from_checkpoint(ckp)
    model.to(DEVICE)
    model = model.eval()

    # load the data
    data_set_sep = SnliDataset(dir=test_dir, nb_sentences=10000, msg=False, keep_neutral=True)
    data_set_sep_less = SnliDataset(dir=test_dir, nb_sentences=10000, msg=False, keep_neutral=True, skeep_sep=True)

    data_loader_sep = DataLoader(data_set_sep, batch_size=4)
    data_loader_sep_less = DataLoader(data_set_sep_less, batch_size=4)

    # evaluation with the SEP token
    with torch.no_grad():
        Y_H = torch.tensor([]).to(DEVICE)
        Y = torch.tensor([]).to(DEVICE)
        for batch in tqdm(data_loader_sep):
            input_ids, attention_mask, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask)
            logits = out["logits"]
            y_hat = torch.argmax(logits, dim=-1)
            Y_H = torch.cat((Y_H, y_hat)).to(DEVICE)
            Y = torch.cat((Y, labels)).to(DEVICE)

    Y = Y.cpu().clone()
    Y_H = Y_H.cpu().clone()

    accuracy = Accuracy()
    print(f">> accuracy with sep : {accuracy(Y.type(torch.uint8), Y_H.type(torch.uint8))}")

    # evaluation with the SEP token
    with torch.no_grad():
        Y_H = torch.tensor([]).to(DEVICE)
        Y = torch.tensor([]).to(DEVICE)
        for batch in tqdm(data_loader_sep_less):
            input_ids, attention_mask, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask)
            logits = out["logits"]
            y_hat = torch.argmax(logits, dim=-1)
            y = torch.argmax(labels, dim=-1)
            Y_H = torch.cat((Y_H, y_hat)).to(DEVICE)
            Y = torch.cat((Y, labels)).to(DEVICE)

    Y = Y.cpu().clone().type(torch.uint8)
    Y_H = Y_H.cpu().clone().type(torch.uint8)

    accuracy = Accuracy()
    print(f">> accuracy without sep : {accuracy(Y, Y_H)}")
