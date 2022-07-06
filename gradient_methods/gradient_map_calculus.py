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
    graph_folder = path.join(plots_folder, "gradient_head")
    if not path.exists(path.join(plots_folder, "gradient_head")):
        os.mkdir(path.join(plots_folder, "gradient_head"))

    return os.getcwd(), graph_folder


def get_gradient(layer: int = 0):
    grad_q = model.bert.encoder.layer[layer].attention.self.query.weight.grad
    grad_k = model.bert.encoder.layer[layer].attention.self.key.weight.grad
    grad_v = model.bert.encoder.layer[layer].attention.self.value.weight.grad

    norm_q = torch.norm(grad_q)
    norm_k = torch.norm(grad_k)
    norm_v = torch.norm(grad_v)

    return torch.norm(torch.tensor([norm_q, norm_k, norm_v]))


# load the data
if __name__ == "__main__":
    # prepare the environment for the
    c, graph_folder = main_prepare()
    os.chdir(c)

    print(">> the git rep : ", end="")
    print(os.getcwd())
    print(f">> the plots location : {graph_folder}")

    ckp = path.join(".cache", "logs", "igrida_trained", "0", "best.ckpt")
    model = BertNliLight.load_from_checkpoint(ckp)
    model = model.train()

    gradient_map = torch.zeros((12, 1)).to(DEVICE)
    model.zero_grad()

    # load the data
    data_set = SnliDataset(dir=test_dir, nb_sentences=10000, msg=False, keep_neutral=True)
    data_loader = DataLoader(data_set, batch_size=4, shuffle=False)
    model.to(DEVICE)

    for batch in tqdm(data_loader):
        sentences, masks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
        output = torch.softmax(model(sentences, masks), dim=-1)
        loss = output[:, torch.argmax(labels)].sum()  # we sum to have a scalar valu to compute the gradient
        loss.backward()
        gradient_map[:, 0] = gradient_map[:, 0] + torch.tensor([get_gradient(layer=l) for l in range(12)]).to(DEVICE)

    print((gradient_map / len(data_set)).cpu().detach().numpy())
    fig = default_plot_colormap((gradient_map / len(data_set)).cpu().detach().numpy(),
                                xlabel="..",
                                ylabel="Layer",
                                title="Gradient MAP")

    plt.savefig(os.path.join(graph_folder, "gradient_map_batch_sum_1.png"))
