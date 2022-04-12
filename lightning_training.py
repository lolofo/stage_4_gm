import argparse

import torch
from torch import nn

import pytorch_lightning as pl


from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import BertModel
from transformers import BertTokenizer

import os

from first_model.lightning_bert_nli import BertNliLight
from custom_data_set import SnliDataset


###############################
### parser for the training ###
###############################

parser = argparse.ArgumentParser()

parser.add_argument("-n" , "--nb_epoch" , type= int)
parser.add_argument("-b" , "--batch_size" , type = int)
parser.add_argument("-t" , "--model_type" , type = int)
parser.add_argument("-d" , "--data_dir")
parser.add_argument("-s" , "--save_dir")
parser.add_argument("-mn" , "--model_name")
parser.add_argument("-nb_train" , "--nb_train_sent" , type = int)
parser.add_argument("-nb_test" , "--nb_test_sent" , type = int)

args = parser.parse_args()

#######################
### epoch and batch ###
#######################

    # --> epoch

n = 1 # default epoch : 1

if args.nb_epoch is not None :
    n = args.nb_epoch

    # --> batch size

batch = 4 # default batch : 4
if args.batch_size is not None :
    batch = args.batch_size


################
### the data ###
################

print("loading the data ...")

data_dir = "data/"

if args.data_dir is not None :
    data_dir = args.data_dir

train_dir = data_dir + "snli_1.0_train.txt"
test_dir = data_dir + "snli_1.0_test.txt"

nb_train = 100
nb_test = 20

if args.nb_train_sent is not None :
    nb_train = args.nb_train_sent

if args.nb_test_sent is not None :
    nb_test = args.nb_test_sent

train_data_set = SnliDataset(dir = train_dir , nb_sentences= nb_train , msg = False)
test_data_set = SnliDataset(dir = test_dir , nb_sentences = nb_test , msg = False)

train_loader = DataLoader(train_data_set, batch_size=4)
val_loader = DataLoader(test_data_set, batch_size=4)

#############
### model ###
#############

model = None

model_type = 1

if args.model_type is not None :   
    model_type = args.model_type

if model_type==1 :
    model = BertNliLight()


######################
### trainer config ###
######################

'''
TODO : - make some research to understand the parameters of the trainer 
       - how to do cpu//gpu training
       - how to get the information of the training (done we do it with the tensorboard)
'''

devices = os.cpu_count()
print("devices : ",devices)

trainer = pl.Trainer(max_epochs = n , devices = devices/2)

#############################
### training of the model ###
#############################


trainer.fit(model, train_loader, val_loader)
    


"""
the commande line :
    python lightning_training.py -n 4 -b 4 -nb_train 5000 -nb_test 1000
"""