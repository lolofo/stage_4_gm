import pandas as pd
import os
import torch

from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os import path

"""
For the inference part we will take into account this dataset
In this dataset we delete all the 
"""

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# different directions

# first we set the direction of the git.
cwd = os.getcwd().split(os.path.sep)
while cwd[-1] != "stage_4_gm":
    os.chdir("..")
    cwd = os.getcwd().split(os.path.sep)

cache = path.join(os.getcwd(), '.cache')
data_dir = path.join(cache, 'raw_data', 'snli_data', 'snli_1.0')

train_dir = path.join(data_dir, 'snli_1.0_train.txt')
dev_dir = path.join(data_dir, 'snli_1.0_dev.txt')
test_dir = path.join(data_dir, 'snli_1.0_test.txt')

# max_pad --> padding for the tokenizer
max_pad = 150

# our one hot encoding for the variable
oh_labels = {'entailment': "[1,0,0]",
             'contradiction': "[0,1,0]",
             'neutral': "[0,0,1]"}


class SnliDataset(Dataset):
    '''
    custom dataset
    '''

    def __init__(self, dir=train_dir,
                 nb_sentences=100,
                 keep_neutral=False,
                 only_label=None,
                 msg=True):
        '''
        initiation of the dataset :
            - the default parameter here are for the training
        '''

        if only_label == "neutral" and not keep_neutral:
            raise Exception("skip and keep error: None dataset production congratulation !")

        buff = pd.read_csv(dir, sep="\t")

        nb_sent = buff.shape[0]

        sentence1 = buff.sentence1
        sentence2 = buff.sentence2
        label = buff.label1

        # some preprocessing
        sentences = []
        labels = []

        for i in range(nb_sent):
            try:
                if keep_neutral:
                    if only_label is not None and label[i] == only_label:
                        sentences.append(sentence1[i] + " [SEP] " + sentence2[i])
                        labels.append(label[i])
                    else:
                        sentences.append(sentence1[i] + " [SEP] " + sentence2[i])
                        labels.append(label[i])

                else:
                    if label[i] != "neutral":
                        if only_label is not None and label[i] == only_label:
                            sentences.append(sentence1[i] + " [SEP] " + sentence2[i])
                            labels.append(label[i])
                        else:
                            sentences.append(sentence1[i] + " [SEP] " + sentence2[i])
                            labels.append(label[i])
            except:
                if msg:
                    print("sent 1 : ", sentence1[i], end="  ===  ")
                    print("sent 2 : ", sentence2[i], end="  ===  ")
                    print(label[i])

        # the datas

        n = min(len(labels), nb_sentences)  # sentences to keep in the data

        t = tokenizer(sentences[0:n], padding="max_length", max_length=max_pad, truncation=True)

        self.ids = t.input_ids
        self.attention_mask = t.attention_mask

        labels = labels[0:n]
        buff = pd.Series(labels)
        # one hot encoding
        self.labels = buff.replace(oh_labels).apply(eval).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = torch.tensor(self.ids[idx])
        att = torch.tensor(self.attention_mask[idx])
        lab = torch.tensor(self.labels[idx])

        # return all the tensors
        return ids, att, lab


if __name__ == "__main__":
    train_data_set = SnliDataset()
    print(len(train_data_set))
    # build a dataloader over this dataset.
    train_dataloader = DataLoader(train_data_set, batch_size=32, shuffle=True)
    print(len(train_dataloader))

    sentences, masks, train_labels = next(iter(train_dataloader))

    print(f"sentence batch shape: {sentences.size()}")
    print(f"attention mask batch shape: {sentences.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    print(torch.max(train_labels, 1)[1].size())

    print(sentences[0, :])
    print(masks[0, :])

    mask = masks[0, :].detach().numpy() == 1
    print(sentences[0, mask])

    print(tokenizer.convert_ids_to_tokens(sentences[0, :]))
    print(tokenizer.convert_ids_to_tokens(sentences[0, mask]))
