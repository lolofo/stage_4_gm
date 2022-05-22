import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BertNli(nn.Module):
    """ Bert modèle for SNLI task

    Attributes :
        bert : a bert-base transformers from the library huggin face
        classifier : a classification head to be able to do the SNLI task

    Methods :
        forward : forward pass of the model
        get_attention : return the attention values in a tensor with an interesting shape
    """

    def __init__(self, freeze_bert=False):

        super(BertNli, self).__init__()

        # bert layer
        # the bert layer will return the layer will return the attention weights
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

        # classifier head
        # dropout >> model.eval() for the test part.
        self.classifier = nn.Sequential(
            # dropout for the cls token
            nn.Dropout(0.1),
            # fully connected layer
            nn.Linear(in_features=768, out_features=3),
            # dropout before the final decision
            nn.Dropout(0.1)
        )

        # do not train bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # access to the attention to make our bert analysis.
        self.bert_output = None

    def forward(self, input_ids, attention_mask, *args, **kwargs):

        self.bert_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     *args, **kwargs
                                     )

        cls_token = self.bert_output.last_hidden_state[:, 0, :]

        # the logits are the weights before the softmax.
        logits = self.classifier(cls_token)

        return logits

    ##############################
    ### function for the study ###
    ##############################

    def get_attention(self,
                      input_ids,
                      attention_mask,
                      test_mod: bool = False,
                      *args, **kwargs):
        """
        :param input_ids :
        :param attention_mask:
        :param test_mod:

        """

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            *args, **kwargs)

        attention_tensor = outputs.attentions

        res = torch.stack(attention_tensor, dim=1)

        # remove the padding tokens

        mask = attention_mask[0, :].detach().numpy() == 1
        res = res[:, :, :, mask, :]
        res = res[:, :, :, :, mask]

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0, mask])

        if test_mod:
            print("test passed : ", end='')
            passed = True
            for n in range(len(attention_tensor)):
                for n_head in range(12):
                    for x in range(input_ids.shape[1]):
                        for y in range(input_ids.shape[1]):
                            if mask[x] == 1 and mask[y] == 1:
                                if attention_tensor[n][0, n_head, x, y] != res[0, n, n_head, x, y]:
                                    passed = False

            if passed:
                print(u'\u2713')
            else:
                print("x")

        return res, tokens, input_ids, attention_mask
