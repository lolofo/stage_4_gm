###############################
### General training script ###
###############################

import pandas as pd
import numpy as np

import sys
# everything will be printed on this file
# we will keep all the informations of the different training into a file
sys.stdout = open('training_informations.txt', 'w')
print("START OF THE TRAINING")

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from transformers import BertTokenizer

# scheduler --> modify the lr through the epochs
from transformers import get_scheduler

from torch.optim import AdamW


from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from first_model.bert_nli import BertNli
from custom_data_set import SnliDataset

import argparse

############################################
### training general parameters (parser) ###
############################################

parser = argparse.ArgumentParser()

parser.add_argument("-n" , "--nb_epoch" , type= int)
parser.add_argument("-b" , "--batch_size" , type = int)
parser.add_argument("-t" , "--model_type" , type = int)
parser.add_argument("-d" , "--data_dir")
parser.add_argument("-s" , "--save_dir")
parser.add_argument("-mn" , "--model_name")

args = parser.parse_args()


#######################
### epoch and batch ###
#######################

    # --> epoch

n = 1 # default epoch : 1

if "nb_epoch" in args :
    n = args.nb_epoch

    # --> batch size

batch = 4 # default batch : 4
if "batch_size" in args :
    batch = args.batch_size

#################
### the model ###
#################

model_type = 1

if "model_type" in args :   
    model_type = args.model_type

snli_model = None

if model_type==1 :
    snli_model = BertNli()

#########################
### device cpu // gpu ###
#########################

device = torch.device("cpu")
print("we will train on the following device : " , device)


#################################
### the data for the training ###
#################################

    # INIT of the train dataloard
    # here we will have 10 000 sentences for the training

data_dir = "data"

if "data_dir" in args :
    data_dir = args.data_dir

train_dir = data_dir + "/snli_1.0_train.txt"
test_dir = data_dir + "/snli_1.0_test.txt"

print("loading data :")
train_data_set = SnliDataset(dir = train_dir , nb_sentences= 1000 , msg = False)
train_data_loader = DataLoader(train_data_set , batch_size = batch , shuffle = True)


test_data_set = SnliDataset(dir = test_dir , nb_sentences = 1000 ,  msg = False)
test_data_loader = DataLoader(test_data_set , batch_size = batch , shuffle = True)

    # make sure that the data load well
    # we print some information about the dataloader

print("\t length data loader : ",len(train_data_loader))

sentences, masks, train_labels = next(iter(train_data_loader))

print(f"\t sentence batch shape: {sentences.size()}")
print(f"\t attention mask batch shape: {sentences.size()}")
print(f"\t Labels batch shape: {train_labels.size()}")


###################################
### loss function and optimizer ###
###################################

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(snli_model.parameters(), lr=5e-5)

lr_scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer, 
    num_warmup_steps=0, 
    num_training_steps=n*len(train_data_loader)
)


#################################################
### training loop (using previous parameters) ###
#################################################

#TODO

def model_training (trainloader , test_data_loader, n = 4) :
    '''
    training loop : generical training loop
    '''


    print()
    print("start training")

    snli_model.to(device)

    for epoch in range(n):  # loop over the dataset multiple times

        print("epoch {}".format(epoch))

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            print("\t batch {}".format(i))
            
            input_ids, attention_mask , labels = data[0].to(device) , data[1].to(device) , data[2].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits = snli_model(input_ids = input_ids , attention_mask = attention_mask)

            # we give the logits to the criterion function
            # the criterion function will 
            loss = criterion(logits,  torch.max(labels, 1)[1])

            loss.backward()
            optimizer.step()

            # update of the learning rate
            lr_scheduler.step()

            # print statistics
            running_loss += loss.item()
            print("\t \t - current loss : {}".format(running_loss/batch))

            
            running_loss = 0.0

    print()
    print("Begining of the test part")
    print()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_data_loader:

            '''
            go through all the test data
            '''

            input_ids, attention_mask , labels = data[0].to(device) , data[1].to(device) , data[2].to(device)

            logits = snli_model(input_ids = input_ids , attention_mask = attention_mask)

            # the class with the highest energy is what we choose as prediction

            labels = torch.max(labels, 1)[1]

            predicted = torch.max(logits, 1)[1]

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test data_set {100 * correct // total} %')

    print()
    print('Finished Training')


    
################
### training ###
################

print("nb epoch" , n)
model_training(train_data_loader , test_data_loader , n)

######################
### save the model ###
######################


save_dir = "checkpoint/"
model_name = "default.pt"

if "save_dir" in args :
    save_dir = args.save_dir

if "model_name" in args :
    model_name = args.model_name


PATH = save_dir +"/"+model_name

torch.save(snli_model.state_dict() , PATH)


#################################
### close the standard output ###
#################################

sys.stdout.close()