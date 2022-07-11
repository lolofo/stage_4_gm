import torch
from tqdm import tqdm
import numpy as np

from training_bert import BertNliLight
from custom_data_set import SnliDataset
from custom_data_set import test_dir, dev_dir
from torch_set_up import DEVICE

import os
from os import path
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt


# set the repository to the git repository
def main_prepare():
    cwd = os.getcwd().split(os.path.sep)
    while cwd[-1] != "stage_4_gm":
        os.chdir("..")
        cwd = os.getcwd().split(os.path.sep)

    # the folder where we will save our data
    plots_folder = os.path.join(os.getcwd(), '.cache', 'plots')
    graph_folder = path.join(plots_folder, "cls_sep")
    if not path.exists(path.join(plots_folder, "cls_sep")):
        os.mkdir(path.join(plots_folder, "cls_sep"))

    return os.getcwd(), graph_folder


if __name__ == "__main__":

    c, graph_folder = main_prepare()
    gradient_map = torch.zeros((150, 13)).to(DEVICE)

    ckp = path.join(".cache", "logs", "igrida_trained", "0", "best.ckpt")
    model = BertNliLight.load_from_checkpoint(ckp)
    model.to(DEVICE)
    model = model.eval()

    x = []
    y = []
    hue = []
    l = ["entailment", "contradiction", "neutral"]

    data_set = SnliDataset(dir=test_dir, nb_sentences=1000, msg=False, keep_neutral=True)
    data_loader = DataLoader(data_set, batch_size=4, shuffle=True)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            sentences, masks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            outputs = model(sentences, masks)
            lb = torch.argmax(labels, dim=-1)  # get the class for every sentences --> have a good plot.
            for i in range(labels.shape[0]):
                for lay in range(13):
                    baseline = outputs["hidden_states"][max(lay - 1, 0)][i, 0, :]
                    buff = outputs["hidden_states"][lay][i, 0, :]
                    x.append(f"layer {lay}")
                    y.append((torch.dot(buff, baseline) / (torch.norm(buff) * torch.norm(baseline))).item())
                    hue.append(l[lb[i]])

    sns.set_theme()
    fig = plt.figure(figsize=(10, 10))
    lay = 0
    x = np.array(x)
    y = np.array(y)
    hue = np.array(hue)

    fig = sns.boxplot(x=x, y=y)

    plt.savefig(os.path.join(graph_folder, "cls_angle_mouvement.png"))
