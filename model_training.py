###############################
### General training script ###
###############################

print("START OF THE TRAINING")

from transformers import get_scheduler

from torch.optim import AdamW

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from first_model.bert_nli import BertNli
from custom_data_set import SnliDataset

import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

############################################
### training general parameters (parser) ###
############################################

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--nb_epoch", type=int)
parser.add_argument("-b", "--batch_size", type=int)
parser.add_argument("-t", "--model_type", type=int)
parser.add_argument("-d", "--data_dir")
parser.add_argument("-s", "--save_dir")
parser.add_argument("-mn", "--model_name")
parser.add_argument("-nb_train", "--nb_train_sent", type=int)
parser.add_argument("-nb_test", "--nb_test_sent", type=int)

args = parser.parse_args()

#######################
### epoch and batch ###
#######################

# --> epoch

n = 2  # default epoch : 2

if args.nb_epoch is not None:
    n = args.nb_epoch

    # --> batch size

batch = 4  # default batch : 4
if args.batch_size is not None:
    batch = args.batch_size

#################
### the model ###
#################

model_type = 1

if args.model_type is not None:
    model_type = args.model_type

snli_model = None

if model_type == 1:
    snli_model = BertNli()
#########################
### device cpu // gpu ###
#########################

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

print("we will train on the following device : ", device)

#################################
### the data for the training ###
#################################

data_dir = "./snli_data/snli_1.0/"

if args.data_dir is not None:
    data_dir = args.data_dir

train_dir = data_dir + "snli_1.0_train.txt"
test_dir = data_dir + "snli_1.0_test.txt"

print("loading data :", end="")

nb_train = 5000
nb_test = 1000

if args.nb_train_sent is not None:
    if nb_train <= 550152:
        nb_train = args.nb_train_sent

if args.nb_test_sent is not None:
    if nb_test <= 10000:
        nb_test = args.nb_test_sent

train_data_set = SnliDataset(dir=train_dir, nb_sentences=nb_train, msg=False)
train_data_loader = DataLoader(train_data_set, batch_size=batch, shuffle=True)

test_data_set = SnliDataset(dir=test_dir, nb_sentences=nb_test, msg=False)
test_data_loader = DataLoader(test_data_set, batch_size=batch, shuffle=True)
print(u'\u2713')

###################################
### loss function and optimizer ###
###################################
criterion = nn.CrossEntropyLoss()

# Adam provides the best performances for the training
optimizer = AdamW(snli_model.parameters(), lr=5e-5)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=n * len(train_data_loader)
)


#################################################
### training loop (using previous parameters) ###
#################################################
def model_training():
    '''
    training loop : generical training loop
    '''

    print()
    print("start training")

    snli_model.to(device)

    for epoch in range(n):

        ## training of the model ##
        snli_model.train()
        with tqdm.tqdm(train_data_loader, unit="batch") as tepoch:
            for data in tepoch:
                input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                logits = snli_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, torch.max(labels, 1)[1])
                loss.backward()
                optimizer.step()

                # update of the learning rate
                lr_scheduler.step()

        ## evaluation on the training set ###
        snli_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        ep = 0
        with torch.no_grad():
            with tqdm.tqdm(train_data_loader, unit="batch") as tepoch:
                for data in tepoch:
                    input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)

                    # write the loss
                    logits = snli_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, torch.max(labels, 1)[1])
                    total_loss += loss.item()
                    ep += 1

                    # write the accuracy
                    labels = torch.max(labels, 1)[1]
                    predicted = torch.max(logits, 1)[1]
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        writer.add_scalar("train_loss", total_loss / ep, epoch)
        writer.add_scalar("train_accuracy", correct / total, epoch)

        ## evaluation on the test set ###
        snli_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        ep = 0
        with torch.no_grad():
            with tqdm.tqdm(test_data_loader, unit="batch") as tepoch:
                for data in tepoch:
                    input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)

                    # write the loss
                    logits = snli_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, torch.max(labels, 1)[1])
                    total_loss += loss.item()
                    ep += 1

                    # write the accuracy
                    labels = torch.max(labels, 1)[1]
                    predicted = torch.max(logits, 1)[1]
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        writer.add_scalar("test_loss", total_loss / ep, epoch)
        writer.add_scalar("test_accuracy", correct / total, epoch)


################
### training ###
################

print("nb epoch", n)
model_training()

######################
### save the model ###
######################


save_dir = "checkpoint"
model_name = "default.pt"

if args.save_dir is not None:
    save_dir = args.save_dir

if args.model_name is not None:
    model_name = args.model_name

PATH = save_dir + "/" + model_name

torch.save(snli_model.state_dict(), PATH)
