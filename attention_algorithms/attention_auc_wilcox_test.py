import numpy as np
import torch
from sklearn import metrics

from attention_algorithms.raw_attention import RawAttention
from attention_algorithms.attention_metrics import normalize_attention


def global_auc_score(sentences,
                     masks,
                     annot_data,
                     model):
    """Calculation of the AUC score sentence by sentence by doing a simple global agregation

    Here for a set of sentences we calculate the auc sentence by sentences and this set of auc is the one
    that we will be used as a target.

    The objective will be then to find other methods to see if we can improve the auc score significantly.
    """
    res = []
    for i in range(sentences.shape[0]):
        j = 0
        sent = sentences[i, :].clone().detach()[None, :]
        mk = masks[i, :].clone().detach()[None, :]
        raw_attention_inst = RawAttention(model=model,
                                          input_ids=sent,
                                          attention_mask=mk,
                                          test_mod=False
                                          )
        try:
            while annot_data["tok_sent"][j] != raw_attention_inst:
                j += 1

            # calculation of the attention score
            as_score = np.zeros(len(raw_attention_inst.tokens))
            for l in range(12):
                for h in range(12):
                    as_score += raw_attention_inst.attention_tensor[0, l, h, :, :].sum(dim=0) \
                        .detach().numpy()

            as_score = normalize_attention(raw_attention_inst.tokens, torch.tensor(as_score)) \
                .detach().numpy()

            # calculation of the AUC for this score
            fpr, tpr, threshold = metrics.roc_curve(annot_data["hg_goal"], as_score)
            res.append(metrics.auc(fpr, tpr))


        except:
            # stop the iteration
            pass
    return res


# calculate the auc_score by delete some layers in our transformer
def layer_del_auc_score(sentences,
                        masks,
                        annot_data,
                        model,
                        layers):
    res = []
    for i in range(sentences.shape[0]):
        j = 0
        sent = sentences[i, :].clone().detach()[None, :]
        mk = masks[i, :].clone().detach()[None, :]
        raw_attention_inst = RawAttention(model=model,
                                          input_ids=sent,
                                          attention_mask=mk,
                                          test_mod=False
                                          )
        try:
            while annot_data["tok_sent"][j] != raw_attention_inst:
                j += 1

            # calculation of the attention score
            as_score = np.zeros(len(raw_attention_inst.tokens))
            for l in layers:
                for h in range(12):
                    as_score += raw_attention_inst.attention_tensor[0, l, h, :, :].sum(dim=0) \
                        .detach().numpy()

            as_score = normalize_attention(raw_attention_inst.tokens, torch.tensor(as_score)) \
                .detach().numpy()

            # calculation of the AUC for this score
            fpr, tpr, threshold = metrics.roc_curve(annot_data["hg_goal"], as_score)
            res.append(metrics.auc(fpr, tpr))


        except:
            # stop the iteration
            pass
    return res


# calculate the auc_score by delete
def tr_del_auc_score(sentences,
                     masks,
                     annot_data,
                     model,
                     TR,
                     auc_map):
    res = []
    for i in range(sentences.shape[0]):
        j = 0
        sent = sentences[i, :].clone().detach()[None, :]
        mk = masks[i, :].clone().detach()[None, :]
        raw_attention_inst = RawAttention(model=model,
                                          input_ids=sent,
                                          attention_mask=mk,
                                          test_mod=False
                                          )
        try:
            while annot_data["tok_sent"][j] != raw_attention_inst:
                j += 1

            # calculation of the attention score
            as_score = np.zeros(len(raw_attention_inst.tokens))
            for l in range(12):
                for h in range(12):
                    if auc_map[l, h] >= TR:
                        as_score += raw_attention_inst.attention_tensor[0, l, h, :, :].sum(dim=0) \
                            .detach().numpy()

            as_score = normalize_attention(raw_attention_inst.tokens, torch.tensor(as_score)) \
                .detach().numpy()

            # calculation of the AUC for this score
            fpr, tpr, threshold = metrics.roc_curve(annot_data["hg_goal"], as_score)
            res.append(metrics.auc(fpr, tpr))


        except:
            # stop the iteration
            pass
    return res
