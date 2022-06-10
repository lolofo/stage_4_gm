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

First of all make sure to use the environnement.

### Virtualenv - pip environment (recommended)

Path to $VENV should be saved in `~/.bashrc`

```commandline
# Specify path to venv
export VENV=path/to/venv
echo $VENV

# Create venv
python -m venv $VENV/bert

# Activate venv
source $VENV/bert/bin/activate

# Replicate on cpu
pip install -r python_env/requirements.cpu.txt --no-cache-dir

# Replicate on gpu
pip install -r python_env/requirements.gpu.txt --no-cache-dir

# Exit venv
deactivate
```

### Virtualenv - conda environment

- if you are using conda you can use the two following command :

```commandline
conda env create -f python_env/environment.yml
conda activate nlp
```

```commandline
conda create --name nlp --file requirements.txt
conda activate nlp
```

**WARNING**: All the environments were exported on windows 11 -64 bits.

### Download the data

To download the snli and e-snli data the command line is the following :

```commandline
python data_download.py
```

All the data downloaded in this part will be stored in the folder : `.cache\raw_data` 


### Pytorch lightning training script

To run the *training_bert.py* for some tests we used the following command line :

```commandline
python training_bert.py --epoch 3 --batch_size 4 --nb_data 16 --experiment bert --version 0

# Or by shorthand
python training_bert.py -e 3 -b 4 -n 16 --experiment bert --version 0
```

The objective was only to see the behaviour of the training with a small amount of data. (Spot some mistakes and see the
behaviour of the loss)

To visualize our training performance we used the tool **tensorboard**. The default logdir in
in `.cache/logs/$EXPERIMENT`
where `$EXPERIMENT` is specified in `--experiment`. The log could be changed using flag `--logdir` or shorthand `-s`

```commandline
tensorboard --logdir .cache/logs/$EXPERIMENT
```
