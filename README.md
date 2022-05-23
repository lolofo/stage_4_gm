
## Table of content

- [Introduction](#introduction)
- [Natural Language Inference (NLI) task](#natural-language-inference-nli-task)
- [The data (SNLI dataset)](#the-data-snli-dataset)
- [Command lines (How to use this git)](#command-lines-how-to-use-this-git)
  - [Pytorch lightning script](#pytorch-lightning-training-script)



## Introduction

## Natural Language Inference (NLI) task

## The data (SNLI dataset)

## Command lines (How to use this git)

First of all make sure to use the environnement provided thanks to the following command lines.

```
conda env create -f python_env/nlp_env.yml
conda activate nlp
```

### Download the data

To download the snli data the command line is the following :
```
python data_download.py
```

### Pytorch lightning training script

To run the *lightning_training.py* for some tests we used the following command line :

```
python lightning_training.py -n 3 -b 4 -nb_train 100 -nb_test 20 -logs log_test
```

The objective was only to see the behaviour of the training with a small amount of data. (Spot some mistakes and see the behaviour of the loss)

To visualize our training performance we used the tool **tensorboard**. If *log_dir* is the name of the foler where there is the logs of your training then you can visualize the performance with the following commande line :

```
tensorboard --logdir log_dir
```
