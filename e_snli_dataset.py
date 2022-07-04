# TODO : e_snli dataset
# input_ids, attention_mask ,annot, label
import pandas as pd
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
from sklearn.utils import shuffle
from modules.transforms import *

from transformers import BertTokenizer
from torch.utils.data import Dataset

DIR = os.path.join(".cache", "raw_data", "e_snli", "cleaned_data")
TRAIN = ["1.csv", "2.csv"]
DEV = ["dev.csv"]
TEST = ["test.csv"]

KEEP_COLS = ["premise", "hypothesis", "label", "hg_goal"]
MAX_PAD = 150

LAB_ENC = {"entailment": 0,
           "neutral": 1,
           "contradiction": 2}


class EsnliDataSet(Dataset):
    def __init__(self, split="TRAIN", nb_data=-1, cache_path=DIR):
        self.dirs = [os.path.join(cache_path, f) for f in eval(split)]  # where the datas are
        self.data = None
        if split == "TRAIN":
            # in the function don't forget to split for the training and validation part.
            df1 = pd.read_csv(self.dirs[0], usecols=KEEP_COLS)
            df2 = pd.read_csv(self.dirs[1], usecols=KEEP_COLS)
            self.data = pd.concat([df1, df2])
        elif split in ["TEST", "DEV"]:
            self.data = pd.read_csv(os.path.join(self.dirs[0]), usecols=KEEP_COLS)

        if nb_data > 0:
            # fraction of data to keep
            frac = nb_data / self.data.shape[0]
            self.data = self.data.sample(frac=frac).reset_index()

    def __getitem__(self, item):
        premise = self.data.premise.values[item]
        hypothesis = self.data.hypothesis.values[item]
        buff = eval(self.data.hg_goal.values[item])
        annotation = buff + [0] * (MAX_PAD - len(buff))
        label = self.data.label.values[item]
        return {"premise": premise, "hypothesis": hypothesis,
                "annotation": torch.tensor(annotation, requires_grad=False),
                "label": torch.tensor(LAB_ENC[label])}

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    print("TEST of the SNLI dataset")

    train_ds = EsnliDataSet(split="TRAIN", nb_data=100)
    print(">> len(train_ds) : ", len(train_ds))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=False)
    d = next(iter(train_dl))
    print(">> shape of the annotation ", d["annotation"].shape)
    print(">> shape of the labels", d["label"].shape)
    print(">> one premise ", d["premise"][0])
    print(">> the corresponding hypothesis", d["hypothesis"][0])
    print(">> some labels", d["label"][:5])
    print(">> labels of the df : ", train_ds.data.label.values[:5])

    print(">> mask comp : ", end="")
    m1 = np.array(eval(train_ds.data.hg_goal.values[0]))
    m2 = d["annotation"][0, :len(m1)].detach().numpy()
    print(m1 == m2)

    # add a test for the tokenized dataset
    # test the different steps of the collate function
    AT = AddSepTransform()
    text = AT(d["premise"], d["hypothesis"])
    print(text[0])

    BT = BertTokenizeTransform()
    input_ids, attention_mask = BT(text)

    print(">> nb_ids : ", sum(attention_mask[0]))
    print(">> len(m1) : ", len(m1))
