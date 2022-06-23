import torch
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

from custom_data_set import SnliDataset, LABELS
from attention_algorithms.raw_attention import RawAttention
from custom_data_set import test_dir, dev_dir
from torch.utils.data import DataLoader
from attention_algorithms.attention_metrics import normalize_attention
from sklearn import metrics


# --> highlight the words and give back a html visu
def hightlight_txt(tokens, attention):
    """
    Build an HTML of text along its weights.
    Args:
        tokens: list of tokens
        attention: list of attention weights
        show_pad: whethere showing padding tokens
    """
    assert len(tokens) == len(attention), f'Length mismatch: f{len(tokens)} vs f{len(attention)}'

    highlighted_text_1 = [f'<span style="background-color:rgba(135,206,250, {weight});">{text}</span>' for weight, text
                          in
                          zip(attention, tokens)]

    return ' '.join(highlighted_text_1)


# --> table's construction
def construct_html_table(metrics_name,
                         annotations,
                         ):
    table = ["<table>"]

    titles = ["<tr>"]
    for t in metrics_name:
        line = f"<th scope=\"col\"> {t} </th>"
        titles.append(line)
    titles.append("</tr>")

    body = []
    for i in range(len(annotations)):
        d = annotations[i]
        body.append(f"<tr>")
        for t in metrics_name:
            body.append(f"<td>{d[t]}</td>")
        body.append("</tr>")

    table += titles + body

    return "".join(table)


# --> html page construction
def construct_html_page_visu(title,
                             table,
                             file_name,
                             H2,
                             paragraph
                             ):
    # create the html file
    if os.path.exists(os.path.join(".cache", "plots", file_name)):
        os.utime(os.path.join(".cache", "plots", file_name))
    else:
        open(os.path.join(".cache", "plots", file_name), "a").close()

    html_page = f"<!DOCTYPE html> <html> <head> <title>{title}</title><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
    style_page = """<style>
    table {
      border-collapse: collapse;
      border-spacing: 0;
      width: 100%;
      border: 1px solid #ddd;
    }

    th, td {
      text-align: left;
      padding: 16px;
    }

    tr:nth-child(even) {
      background-color: #f2f2f2;
    }
    </style>
    </head>
    """
    html_page += style_page

    body = f"<body><h2>{H2}</h2>"

    paragraph = f"<p>{paragraph}</p>"

    html_table = f"{table} </body></html>"
    body += paragraph + html_table

    html_page += body
    # write into the html wile
    with open(os.path.join(".cache", "plots", file_name), 'w') as f:
        f.write(html_page)


# --> creation of an html_page for the AUC problem
def auc_html_page(model,
                  dir,
                  auc_map,
                  page_title, page_file_name):
    # the data we annotated.
    x = []  # --> the normal AUC
    y = []  # --> the AUC for the weighted version

    e_snli_data = pd.read_csv(os.path.join('.cache', 'raw_data', 'e_snli', 'cleaned_data', f'{dir}.csv'), sep=",") \
        [["tok_sent", "hg_goal", "label"]]
    e_snli_data.head()

    # load the data

    data_set = SnliDataset(dir=eval(f"{dir}_dir"), nb_sentences=1000, msg=False)
    data_loader = DataLoader(data_set, batch_size=1000, shuffle=True)
    sentences, masks, train_labels = next(iter(data_loader))

    # >> where we will store the annotation
    annotations = []

    for i in range(sentences.shape[0]):
        # >> init of the dictionary
        buff_dict = {
            "ESNLI(s)": None,
            "SUM AS(s)": None,
            "W SUM AS(s)": None,
            "PRED LAB": None,
            "REAL LAB": None,
            "AUC(AS(s), ESNLI)": 0,
            "AUC(sum(w*AS(s), ESNLI))": 0
        }

        sent = sentences[i, :].clone().detach()[None, :]
        mk = masks[i, :].clone().detach()[None, :]
        raw_attention_inst = RawAttention(model=model,
                                          input_ids=sent,
                                          attention_mask=mk,
                                          test_mod=False
                                          )

        # >> prediction
        pred = torch.argmax(model(sent, mk), dim=1)
        buff_dict["PRED LAB"] = LABELS[pred]

        # the attention_score of the heads
        # we compute the sum of all the scores.
        AS_sent = np.zeros(len(raw_attention_inst.tokens))
        AS_sent_w = np.zeros(len(raw_attention_inst.tokens))

        # we combine all the different annotations
        for l in range(12):
            for h in range(12):
                AS_sent += raw_attention_inst.attention_tensor[0, l, h, :, :].sum(dim=0).detach().numpy()
                AS_sent_w += auc_map[l, h] * raw_attention_inst.attention_tensor[0, l, h, :, :].sum(
                    dim=0).detach().numpy()

        # >> we normalize the attention weights >> get rid of the special tokens
        AS_sent = normalize_attention(raw_attention_inst.tokens, torch.tensor(AS_sent)).detach().numpy()
        AS_sent_w = normalize_attention(raw_attention_inst.tokens, torch.tensor(AS_sent_w)).detach().numpy()

        try:
            j = 0
            while eval(e_snli_data["tok_sent"][j]) != raw_attention_inst.tokens:
                j += 1

            # >> the annotations
            vis = hightlight_txt(raw_attention_inst.tokens, AS_sent / 0.1)
            buff_dict["SUM AS(s)"] = vis

            vis = hightlight_txt(raw_attention_inst.tokens, torch.tensor(eval(e_snli_data["hg_goal"][j])))
            buff_dict["ESNLI(s)"] = vis

            vis = hightlight_txt(raw_attention_inst.tokens, AS_sent_w / 0.1)
            buff_dict["W SUM AS(s)"] = vis

            # >> the auc score
            fpr, tpr, threshold = metrics.roc_curve(eval(e_snli_data["hg_goal"][j]), AS_sent)
            auc_score = metrics.auc(fpr, tpr)
            x.append(auc_score)
            vis = f"{np.round(auc_score, 3) * 100}"
            buff_dict["AUC(AS(s), ESNLI)"] = vis

            # >> the wieghted auc score
            fpr, tpr, threshold = metrics.roc_curve(eval(e_snli_data["hg_goal"][j]), AS_sent_w)
            auc_score = metrics.auc(fpr, tpr)
            y.append(auc_score)
            vis = f"{np.round(auc_score, 3) * 100}"
            buff_dict["AUC(sum(w*AS(s), ESNLI))"] = vis

            buff_dict["REAL LAB"] = e_snli_data["label"][j]

            annotations.append(buff_dict)
        except Exception as e:
            print(e)

    table = construct_html_table(list(buff_dict.keys()),
                                 annotations)
    U, p = mannwhitneyu(x, y)

    paragraph = """
    We will do the Wilcoxon-Mann-Whitney test between the last two columns of the table below to see if there is
    a (significant) difference between the last two columns. <br>
    """
    paragraph += f"- Wilcoxon-Mann-Whitney statistique {U} <br> - Wilcoxon-Mann-Whitney p-value {np.round(p, 3)}"

    construct_html_page_visu(title=page_title,
                             table=table,
                             file_name=page_file_name,
                             H2="Comparison of the global and the weighted sum",
                             paragraph=paragraph
                             )
