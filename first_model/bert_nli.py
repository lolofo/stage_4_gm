import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer

'''
for the tokenizer :
    101 --> [CLS] (begining of a sentence)
    102 --> [SEP] (end of a sentence)
'''

'''
modification to make :
    - the model must not take as input the sentences
    - the model must take as input, numerical data --> transformation of the data

A torch model always take as input,  numerical data.

For the data preparation we need to pad all the sentences
'''


class BertNli(nn.Module):
    '''
    Bert the natural language inference task :
        s1 [SEP] s2 ==> three possible classes :

            - entailment (1)
            - contradiction (2)
            - neutral (3)

    we need to prepare the data set thanks to these parameters
    '''

    def __init__(self, freeze_bert=False):

        super(BertNli, self).__init__()

        # bert layer
        # the bert layer will return the layer will return the attention weights
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

        # classifier head
        self.classifier = nn.Sequential(
            # fully connected layer
            nn.Linear(in_features=768, out_features=3),

        )

        # do not train bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # access to the attention to make our bert analysis.
        self.bert_output = None

    def forward(self, input_ids, attention_mask, *args, **kwargs):

        '''
        input_ids : torch.tensor of shape (batch_size , max_pad)
        attention_mask : torch.tensor of shape (batch_size , max_pad)
        '''

        self.bert_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     *args, **kwargs
                                     )

        cls_token = self.bert_output.last_hidden_state[:, 0, :]

        # the logits are the weights before the softmax.
        logits = self.classifier(cls_token)

        return logits
