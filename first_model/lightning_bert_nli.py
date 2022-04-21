
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from torch.optim import AdamW


from transformers import BertModel
from transformers import BertTokenizer

from torchmetrics.classification import Accuracy

criterion = nn.CrossEntropyLoss()

"""
TODO : create a function to access to the hidden attention state and make a good visualization of it.
"""

class BertNliLight(pl.LightningModule):

    def __init__(self , freeze_bert = False) :

        super().__init__()

        # bert layer
        # the bert layer will return the layer will return the attention weights
        self.bert = BertModel.from_pretrained('bert-base-uncased' , output_attentions = True)

        # classifier head
        self.classifier = nn.Sequential(
            # fully connected layer
            nn.Linear(in_features = 768 , out_features = 3),

        )

        self.accuracy = Accuracy(num_class = 3)



    def forward (self , input_ids , attention_mask , *args , **kwargs) :

        '''
        input_ids : torch.tensor of shape (batch_size , max_pad)
        attention_mask : torch.tensor of shape (batch_size , max_pad)
        '''
        
        self.bert_output = self.bert(input_ids = input_ids , 
                        attention_mask = attention_mask , 
                        *args , **kwargs
                        )

        cls_token = self.bert_output.last_hidden_state[:,0,:]

        # the logits are the weights before the softmax.
        logits = self.classifier(cls_token)

        return(logits)
    
    def configure_optimizers(self):
        '''
        define the optimizer for the training
        '''
        optimizer = AdamW(self.parameters(), lr=5e-5)

        return optimizer
        
    
    def training_step(self, train_batch, batch_idx):
        
        input_ids, attention_mask , labels = train_batch
        logits = self.forward(input_ids , attention_mask)
        loss = criterion(logits , torch.max(labels, 1)[1])
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, val_batch, batch_idx):

        
        input_ids, attention_mask , labels = val_batch
        logits = self.forward(input_ids , attention_mask)

        # calculation of the loss
        loss = criterion(logits , torch.max(labels, 1)[1])
        self.log('val_loss', loss)

        # some tools for the end_validation
        class_pred = torch.max(logits , 1)[1]
        class_true = torch.max(labels, 1)[1]

        return {'loss' : loss , 'preds' : class_pred , 'target' : class_true}


    def validation_step_end(self, outputs):
        
        self.accuracy(outputs['preds'], outputs['target'])
        self.log('metric', self.accuracy)