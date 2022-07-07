import torch
from tqdm import tqdm

from training_bert import BertNliLight
from custom_data_set import SnliDataset
from custom_data_set import test_dir, dev_dir
from torch.utils.data import DataLoader

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


def get_cls_max_gradient(layer: int = 0):
    pass


# load the data
if __name__ == "__main__":
    c, graph_folder = main_prepare()
    gradient_map = torch.zeros((150, 13)).to(DEVICE)

    ckp = path.join(".cache", "logs", "igrida_trained", "0", "best.ckpt")
    model = BertNliLight.load_from_checkpoint(ckp)
    model.to(DEVICE)
    model = model.train()

    data_set = SnliDataset(dir=test_dir, nb_sentences=1000, msg=False, keep_neutral=True)
    data_loader = DataLoader(data_set, batch_size=4, shuffle=True)

    for batch in tqdm(data_loader):
        model.zero_grad()
        sentences, masks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
        outputs = model(sentences, masks)

        layers = []
        for i in range(13):
            layers.append(outputs["hidden_states"][i])
            layers[i].retain_grad()

        torch.softmax(outputs["logits"][:, torch.argmax(labels)], dim=-1).sum().backward()

        for i in range(13):
            buff = layers[i].grad.sum(dim=0).max(dim=-1)[0]  # sum over the batch
            gradient_map[:, i] = gradient_map[:, i] + buff

    fig = default_plot_colormap(gradient_map.cpu().detach().numpy()[1:50, :],
                                xlabel="layer",
                                ylabel="token_saillance",
                                title="Gradient MAP",
                                show_values=False)

    plt.savefig(os.path.join(graph_folder, "gradient_cls_map_cls_less.png"))
