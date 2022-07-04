import torch
import os
from transformers import BertTokenizer
import pandas as pd

tk = BertTokenizer.from_pretrained('bert-base-uncased')


def _sent_tokenize(sent: list):
    """
    tokenize a sentence and get it's score
    """
    buff = sent.copy()
    tok_res = []
    hg_res = []
    for w in buff:
        # if the word is higlighted
        t = None
        if "*" in w:
            # the word is higlighted
            t = tk(w.replace("*", "")).input_ids
            # remove the special tokens
            _ = t.pop(0)
            _ = t.pop(-1)

            hg_res += [1] * len(t)
        else:
            t = tk(w).input_ids
            _ = t.pop(0)
            _ = t.pop(-1)

            hg_res += [0] * len(t)

        tok_res += t

    return list(tk.convert_ids_to_tokens(torch.tensor(tok_res).detach().numpy())), \
           hg_res


def _reformat_csv(data: pd.DataFrame):
    """
    Remove unecessary columns, rename columns for better understanding. Notice that we also remove extra explanation
    columns.
    Args: data (pandas.DataFrame): Original data given by eSNLI dataset

    Returns:
        (pandas.DataFrame) clean data
    """

    rename_cols = {
        'Sentence1': 'premise',
        'Sentence2': 'hypothesis',
        'gold_label': 'label',
        'Explanation_1': 'explanation',
        'Sentence1_marked_1': 'highlight_premise',
        'Sentence2_marked_1': 'highlight_hypothesis'
    }

    drop_cols = ['pairID', 'WorkerId'
                           'Sentence1_Highlighted_1', 'Sentence2_Highlighted_1',
                 'Explanation_2', 'Sentence1_marked_2', 'Sentence2_marked_2',
                 'Sentence1_Highlighted_2', 'Sentence2_Highlighted_2',
                 'Explanation_3', 'Sentence1_marked_3', 'Sentence2_marked_3',
                 'Sentence1_Highlighted_3', 'Sentence2_Highlighted_3']

    if data.isnull().values.any():
        data = data.dropna().reset_index()

    # rename column
    data = data.rename(
        columns=rename_cols
        # drop unneeded
    ).drop(
        columns=drop_cols, errors='ignore'
    )[['premise', 'hypothesis', 'label', 'explanation', 'highlight_premise', 'highlight_hypothesis']]

    def correct_quote(txt, hl):
        """
        Find the incoherent part in text and replace the corresponding in highlight part
        """

        # find different position between the 2
        diff = [i for i, (l, r) in enumerate(zip(txt, hl.replace('*', ''))) if l != r]
        # convert into list to be able to modify character
        txt, hl = list(txt), list(hl)
        idx = 0
        for pos_c, c in enumerate(hl):
            if c == '*': continue
            if idx in diff: hl[pos_c] = txt[idx]
            idx += 1

        hl = ''.join(hl)
        return hl

    # correct some error
    for side in ['premise', 'hypothesis']:
        data[side] = data[side].str.strip() \
            .str.replace('\\', '', regex=False) \
            .str.replace('*', '', regex=False)
        data[f'highlight_{side}'] = data[f'highlight_{side}'] \
            .str.strip() \
            .str.replace('\\', '', regex=False) \
            .str.replace('**', '*', regex=False)  # only one highlight

        # replace all the simple quote (') by double quote (") as orignal phrases
        idx_incoherent = data[side] != data[f'highlight_{side}'].str.replace('*', '', regex=False)
        sub_data = data[idx_incoherent]
        replacement_hl = [correct_quote(txt, hl) for txt, hl in
                          zip(sub_data[side].tolist(), sub_data[f'highlight_{side}'].tolist())]
        data.loc[idx_incoherent, f'highlight_{side}'] = replacement_hl

    # add some tokenization to the dataset
    for side in ['premise', 'hypothesis']:
        new_col_names = [f"tok_{side}", f"goal_{side}"]
        buff_1 = []
        buff_2 = []

        for i in range(data.shape[0]):
            sent = data[f"highlight_{side}"].values[i].split(" ")
            tok_res, hg_res = _sent_tokenize(sent)
            buff_1.append(tok_res)
            buff_2.append(hg_res)

        # add the columns to the dataframe
        data[new_col_names[0]] = buff_1.copy()
        data[new_col_names[1]] = buff_2.copy()

    return data


def _combine_sentences(data: pd.DataFrame):
    data['CLS'] = [["[CLS]"]] * data.shape[0]
    data['SEP'] = [["[SEP]"]] * data.shape[0]
    data['BUFF'] = [[0]] * data.shape[0]

    data['tok_sent'] = data['CLS'] + data['tok_premise'] + data['SEP'] + data['tok_hypothesis'] + data['SEP']
    data['hg_goal'] = data['BUFF'] + data['goal_premise'] + data['BUFF'] + data['goal_hypothesis'] + data['BUFF']

    drop_columns = ['tok_premise', 'goal_premise',
                    'tok_hypothesis', 'goal_hypothesis',
                    'CLS', 'BUFF', 'SEP']

    return data

def download_e_snli_data():
    cwd = os.getcwd().split(os.path.sep)
    while cwd[-1] != "stage_4_gm":
        os.chdir("..")
        cwd = os.getcwd().split(os.path.sep)

    directory = os.path.join(os.getcwd(), ".cache", "raw_data", "e_snli")
    folders = ["esnli_" + u for u in ["dev.csv", "test.csv", "train_1.csv", "train_2.csv"]]
    dirs = [os.path.join(directory, f) for f in folders]
    save_dir = os.path.join(directory, "cleaned_data")

    if not (os.path.exists(save_dir)):
        os.mkdir(save_dir)

    for d in dirs:
        df = pd.read_csv(d, sep=",")
        df = _reformat_csv(df)
        df = _combine_sentences(df)
        df.to_csv(os.path.join(save_dir, d.split("_")[-1]))

if __name__ == "__main__":
    download_e_snli_data()
