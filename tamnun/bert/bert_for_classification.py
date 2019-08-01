from pytorch_transformers import BertModel
from torch import nn


class BertForClassification(nn.Module):
    """
    A pytorch module for Text Classification using BERT.
    """

    def __init__(self, output_size, bert_model_name, dropout=0.0, hidden_size=768, bert_initial_model=None):
        super(BertForClassification, self).__init__()

        if bert_initial_model is not None:
            self.bert = bert_initial_model
        else:
            self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, attention_mask=None):
        _, bert_pooled = self.bert(input_ids, attention_mask=input_ids > 0)
        dropout_output = self.dropout(bert_pooled)
        liner_output = self.linear(dropout_output)
        return liner_output