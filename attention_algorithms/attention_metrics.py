#####################################
### import the different packages ###
#####################################

import sklearn.metrics as metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
from attention_algorithms.raw_attention import RawAttention
import tqdm


###############################
### normalize the attention ###
###############################

# --> make a standard normalization of the attention weights
def normalize_attention(tokens, attention):
    assert len(tokens) == len(attention), f'Length mismatch: f{len(tokens)} vs f{len(attention)}'

    buff = []
    for i in range(len(attention)):
        if tokens[i] not in SPECIAL_TOKENS:
            buff.append(float(attention[i].detach().numpy()))
        else:
            # we now that there will be no attention on the special tokens
            buff.append(0)
    buff = torch.tensor(buff)

    w_min, w_max = torch.min(buff), torch.max(buff)

    # In case of uniform: we do the normalization and after we delete the special tokens
    if w_min == w_max:
        w_min = 0.
    buff = (buff - w_min) / (w_max - w_min)

    return buff


# --> make a softmax normalization of the attention weights
#     not really good because we have a normalization and it sums to one for a sentence.
def softmax_normalization(tokens, attention: torch.tensor):
    assert len(tokens) == len(attention), f'Length mismatch: f{len(tokens)} vs f{len(attention)}'

    sft_max_norm = torch.nn.Softmax(dim=0)
    w_norm = sft_max_norm(attention)

    buff = []
    for i in range(len(attention)):
        if tokens[i] not in SPECIAL_TOKENS:
            buff.append(float(w_norm[i].detach().numpy()))
        else:
            buff.append(0)
    buff = torch.tensor(buff)

    return buff


#####################################
### end of noramlization function ###
#####################################

########################################################################################################################
########################################################################################################################
########################################################################################################################

###############################
### default plot functions ###
###############################

def default_plot_colormap(map,
                          xlabel, ylabel, title,
                          xstick=None,
                          sz=(10, 10)):
    # the global figure
    fig = plt.figure(figsize=sz)
    plt.imshow(map, aspect='auto', cmap='Purples')
    plt.title(title)
    ax = plt.gca()

    # the x-axis
    plt.xlabel(xlabel)
    ax.set_xticks(range(map.shape[1]))
    if xstick is None:
        x_label_list = [str(i) for i in range(map.shape[1])]
        ax.set_xticklabels(x_label_list)
    else:
        ax.set_xticklabels(xstick)

    # the y-axis
    plt.ylabel(ylabel)
    ax.set_yticks(range(map.shape[0]))
    y_label_list = [str(i + 1) for i in range(map.shape[0])]
    ax.set_yticklabels(y_label_list)

    # for each cell
    for x_index in range(map.shape[1]):
        for y_index in range(map.shape[0]):
            label = None
            if type(map[y_index, x_index]) == np.bool_:
                label = str(map[y_index, x_index])
            else:
                label = str(np.round(map[y_index, x_index], 3))
            ax.text(x_index, y_index, label, color='black', ha='center', va='center')

    plt.grid()
    plt.colorbar()

    return fig


####################################
### end of default plot function ###
####################################

########################################################################################################################
########################################################################################################################
########################################################################################################################

#################
### AUC study ###
#################

# --> plot the roc_curve of the model
def plot_roc_curve(Y_test, probs):
    preds = probs
    fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    fig = plt.figure(figsize=(10, 10))
    plt.title('ROC CURVE')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

    plt.plot([0, 1], [0, 1], 'r--', label="random classifier")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right', prop={'size': 20})

    # selection of the best threshold

    best_tr = threshold[np.argmax(tpr - fpr)]

    return fig, best_tr


# --> multiple roc curves plots
def combine_roc_curves(Y_test, probs,
                       legend=["max_agreg", "avg_agreg"]):
    fig = plt.figure(figsize=(10, 10))
    plt.title('ROC CURVE')
    plt.plot([0, 1], [0, 1], 'r--', label="random classifier")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    assert len(probs) == len(legend)

    for i in range(len(probs)):
        preds = probs[i]
        fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('ROC CURVE')
        plt.plot(fpr, tpr, label=f"AUC {legend[i]} : {np.round(roc_auc, 2)}")
        plt.legend(loc='lower right', prop={'size': 10})

    return fig

########################
### end of AUC study ###
########################

########################################################################################################################
########################################################################################################################
########################################################################################################################

##########################################
### The attention score for each layer ###
##########################################


# some constants
class LenException(Exception):
    pass


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]"]


# --> calculate the attention score
def attention_score(sentences, masks,
                    e_snli_data,
                    model,
                    TR_q: float = 0.5,
                    quantiles_calc: bool = True):
    """ attention_score function

    :param sentences: the tokenized sentences
    :param masks: attention masks for the sentences
    :param e_snli_data: dataframe for the annotation
    :param model: snli type bert model
    :param TR_q: the quantile we should consider for the threshold
    :param quantiles_calc: should we calculate the different thresold

    return the attention map as a dictionnary, and the e-snli annotation value
    """
    Y_test = []
    nb_err = 0
    # where we will store the attention
    pur_attention = {}

    # the quantiles vector for our threshold
    quantiles = {}

    for i in range(12):
        pur_attention[f"layer_{i}"] = {}
        quantiles[f"layer_{i}"] = {}
        for j in range(12):
            pur_attention[f"layer_{i}"][f"head_{j}"] = []
            quantiles[f"layer_{i}"][f"head_{j}"] = []

    nb_it = sentences.shape[0]
    print(f">> start the calculus for {nb_it} sentences")
    for _, i in enumerate(tqdm.tqdm(range(nb_it))):
        j = 0
        # iteration through all the sentences
        # construct the raw attention object
        sent = sentences[i, :].clone().detach()[None, :]
        mk = masks[i, :].clone().detach()[None, :]
        raw_attention_inst = RawAttention(model=model,
                                          input_ids=sent,
                                          attention_mask=mk,
                                          test_mod=False
                                          )

        # find the e-snli sentence that corresponds to our problem
        try:
            while j < e_snli_data.shape[0] and raw_attention_inst.tokens != eval(e_snli_data["tok_sent"][j]):
                j += 1

            if j >= e_snli_data.shape[0]:
                raise LenException

            if raw_attention_inst.tokens != eval(e_snli_data["tok_sent"][j]):
                raise LenException

            else:
                # once the sentence is found >> add the e-snli annotation
                Y_test += eval(e_snli_data["hg_goal"][j])

                # loop over every head of every layer
                for l in range(12):
                    for h in range(12):
                        mat = raw_attention_inst.attention_tensor[0, l, h, :, :]
                        # make the sum on the column >> agregation of the weights
                        # >> dim=0 >> we reduce the number of lines
                        b = mat.sum(dim=0)
                        # >> remove the special tokens
                        # >> normalize the attention
                        b = normalize_attention(raw_attention_inst.tokens, b)
                        # >> add the attention
                        pur_attention[f"layer_{l}"][f"head_{h}"] += list(b.detach().numpy())

                        # >> calculate the quantile object
                        if quantiles_calc:
                            # update the quantiles
                            # quantiles for every layers
                            quantiles[f"layer_{l}"][f"head_{h}"] += list(
                                np.repeat(np.quantile(b.detach().numpy(), TR_q),
                                          repeats=len(eval(e_snli_data["hg_goal"][j]))))

        except LenException:
            # count the different errors
            nb_err += 1

    # >> the errors are for the sentences we didn't found in the snli dataset
    print(f">> nb_errors : {nb_err}")

    # how many labeled example do we have
    print(f">> len Y_test : {len(Y_test)}")

    # return the two objects to calculate the metric for each head.
    return pur_attention, Y_test, quantiles


###############################
### Precision recall metric ###
###############################

# problem impossible to use the torch metric
# we don't have the same treshold for our sentences

from sklearn.metrics import precision_score


def precision_recall_map(sentences, masks,
                         e_snli_data,
                         TR_q,
                         model):
    m = np.zeros((12, 12))
    pur_attention, Y_test, quantiles = attention_score(sentences=sentences, masks=masks,
                                                       e_snli_data=e_snli_data,
                                                       model=model,
                                                       TR_q=TR_q,
                                                       quantiles_calc=True)

    Y_test = np.array(Y_test)

    for l in range(12):
        for h in range(12):
            preds = np.array(np.array(pur_attention[f"layer_{l}"][f"head_{h}"]) >= \
                             np.array(quantiles[f"layer_{l}"][f"head_{h}"]),
                             dtype=int)

            m[l, h] = precision_score(Y_test, preds)
    return pur_attention, m


########################################################################################################################
########################################################################################################################
########################################################################################################################

#############################
### Cross attention Study ###
#############################

def construct_cross_mask(mask, sep_idx):
    buff = mask.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if i < sep_idx < j:
                buff[i, j] = 1

            if i > sep_idx > j:
                buff[i, j] = 1
    return buff


def cross_attention_score(sentences, masks,
                          e_snli_data,
                          model):
    """ attention_score function

    :param sentences: the tokenized sentences
    :param masks: attention masks for the sentences
    :param e_snli_data: dataframe for the annotation
    :param model: snli type bert model

    return the attention map as a dictionnary, and the e-snli annotation value
    """
    Y_test = []
    nb_err = 0
    # where we will store the attention
    pur_attention = {}

    # the quantiles vector for our threshold
    quantiles = {}

    for i in range(12):
        pur_attention[f"layer_{i}"] = {}
        quantiles[f"layer_{i}"] = {}
        for j in range(12):
            pur_attention[f"layer_{i}"][f"head_{j}"] = []
            quantiles[f"layer_{i}"][f"head_{j}"] = []

    nb_it = sentences.shape[0]
    print(f">> start the calculus for {nb_it} sentences")
    for _, i in enumerate(tqdm.tqdm(range(nb_it))):
        j = 0
        # iteration through all the sentences
        # construct the raw attention object
        sent = sentences[i, :].clone().detach()[None, :]
        mk = masks[i, :].clone().detach()[None, :]
        raw_attention_inst = RawAttention(model=model,
                                          input_ids=sent,
                                          attention_mask=mk,
                                          test_mod=False
                                          )

        # find the e-snli sentence that corresponds to our problem
        try:
            while j < e_snli_data.shape[0] and raw_attention_inst.tokens != eval(e_snli_data["tok_sent"][j]):
                j += 1

            if j >= e_snli_data.shape[0]:
                raise LenException

            if raw_attention_inst.tokens != eval(e_snli_data["tok_sent"][j]):
                raise LenException

            else:
                # once the sentence is found >> add the e-snli annotation
                Y_test += eval(e_snli_data["hg_goal"][j])

                AS_lh_s = np.zeros(len(raw_attention_inst.tokens))

                # >> where do we have the token [SEP]
                sep_pos = raw_attention_inst.tokens.index("[SEP]")
                cross_mask = np.zeros((len(raw_attention_inst.tokens), len(raw_attention_inst.tokens)))
                cross_mask = construct_cross_mask(cross_mask, sep_pos)

                for l in range(12):
                    for h in range(12):
                        mat = torch.tensor(raw_attention_inst.attention_tensor[0, l, h, :, :] \
                                           .detach().numpy() * cross_mask)
                        # make the sum on the column >> agregation of the weights
                        # >> dim=0 >> we reduce the number of lines
                        b = mat.sum(dim=0)
                        # >> remove the special tokens
                        # >> normalize the attention
                        b = normalize_attention(raw_attention_inst.tokens, b)
                        # >> add the attention
                        pur_attention[f"layer_{l}"][f"head_{h}"] += list(b.detach().numpy())

        except LenException:
            # count the different errors
            nb_err += 1

    # >> the errors are for the sentences we didn't found in the snli dataset
    print(f">> nb_errors : {nb_err}")

    # how many labeled example do we have
    print(f">> len Y_test : {len(Y_test)}")

    # return the two objects to calculate the metric for each head.
    return pur_attention, Y_test, quantiles


########################################################################################################################
########################################################################################################################
########################################################################################################################

#################
### CLS study ###
#################


def cls_attention_score(sentences, masks,
                        e_snli_data,
                        model):
    """ attention_score function

    :param sentences: the tokenized sentences
    :param masks: attention masks for the sentences
    :param e_snli_data: dataframe for the annotation
    :param model: snli type bert model
    :param TR_q: the quantile we should consider for the threshold
    :param quantiles_calc: should we calculate the different thresold

    return the attention map as a dictionnary, and the e-snli annotation value
    """
    Y_test = []
    nb_err = 0
    # where we will store the attention
    pur_attention = {}

    # the quantiles vector for our threshold
    quantiles = {}

    for i in range(12):
        pur_attention[f"layer_{i}"] = {}
        quantiles[f"layer_{i}"] = {}
        for j in range(12):
            pur_attention[f"layer_{i}"][f"head_{j}"] = []
            quantiles[f"layer_{i}"][f"head_{j}"] = []

    nb_it = sentences.shape[0]
    print(f">> start the calculus for {nb_it} sentences")
    for _, i in enumerate(tqdm.tqdm(range(nb_it))):
        j = 0
        # iteration through all the sentences
        # construct the raw attention object
        sent = sentences[i, :].clone().detach()[None, :]
        mk = masks[i, :].clone().detach()[None, :]
        raw_attention_inst = RawAttention(model=model,
                                          input_ids=sent,
                                          attention_mask=mk,
                                          test_mod=False
                                          )

        # find the e-snli sentence that corresponds to our problem
        try:
            while j < e_snli_data.shape[0] and raw_attention_inst.tokens != eval(e_snli_data["tok_sent"][j]):
                j += 1

            if j >= e_snli_data.shape[0]:
                raise LenException

            if raw_attention_inst.tokens != eval(e_snli_data["tok_sent"][j]):
                raise LenException

            else:
                # once the sentence is found >> add the e-snli annotation
                Y_test += eval(e_snli_data["hg_goal"][j])

                # loop over every head of every layer
                for l in range(12):
                    for h in range(12):
                        b = raw_attention_inst.attention_tensor[0, l, h, 0, :].detach().numpy()
                        b = [x if x not in SPECIAL_TOKENS else 0 for x in b]
                        # >> add the attention
                        pur_attention[f"layer_{l}"][f"head_{h}"] += list(np.array(b))

        except LenException:
            pass

    # return the two objects to calculate the metric for each head.
    return pur_attention, Y_test
