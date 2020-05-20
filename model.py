import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel


class BertComplaint2Product(nn.Module):
    """ BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the average of the final hidden states for the entire sentence. """

    def __init__(self, bert_weights='bert-base-uncased', num_labels=30, freeze_bert = False, hidden_dropout_prob=0.0):
        super(BertComplaint2Product, self).__init__()

        self.bert_weights = bert_weights
        self.num_labels = num_labels
        self.freeze_bert = freeze_bert
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = 768

        # Instanciate pretrained Bert
        self.bert = BertModel.from_pretrained(self.bert_weights)

        # Freeze Bert layers
        if self.freeze_bert:
          for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = torch.nn.Dropout(self.hidden_dropout_prob)

        self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                           token_type_ids=token_type_ids)

        # First element of the output is the last hidden layer containing the 
        # Hidden states of the entire sentence
        # Get more information about the text by doing the average of the hidden states of the last layer
        # Here depending on the task we could also consider averaging with more hidden layers of Bert.
        # In ablation study of Bert article it's shown that averaging or concatenating 
        # on last four hidden layers of Bert gives the best results on a NER task.

        last_hidden_layer = output[0]                                               # shape [16, 256, 768]  # [Batch, max_seq_len, hidden_size]
        last_hidden_avg = torch.mean(last_hidden_layer, dim=1)                      # shape [16, 768]
          
        dropout_output = self.dropout(last_hidden_avg)

        proba_product = F.softmax(self.classifier(dropout_output), dim=1)           # shape [16, 30]

        return proba_product.float()
