import torch
from pytorch_transformers import AdamW
from tamnun.bert.bert_for_classification import BertForClassification
from tamnun.bert.constants import BERT_BASE_UNCASED_MODEL_NAME
from tamnun.core import TorchEstimator

EPOCHS = 5
BATCH_SIZE = 4

class BertClassifier(TorchEstimator):
    """ A sklearn like class for text classification based on BERT.
    This class uses BERT and a linear layer on top of it to perform binary/multi-class classification. The loss function
    is determined automatically by the shape of the target variable.
    """
    def __init__(self, num_of_classes, bert_model_name=BERT_BASE_UNCASED_MODEL_NAME, verbose=True,
                 bert_initial_model=None, lr=1e-3):
        """
        bert_model:  The name of the BERT model to use (default is 'bert-base-multilingual-cased'). Accepts one if the values from here: https://github.com/huggingface/pytorch-pretrained-BERT/blob/98dc30b21e3df6528d0dd17f0910ffea12bc0f33/pytorch_pretrained_bert/modeling.py#L36
        verbose:  If True (by default, outputs the training progress (epochs, steps and loss)
        bert_initial_model: A pytroch implementation for BERT, will be used as base model instead of the one provided in "bert_model"
        """
        module = BertForClassification(num_of_classes,
                                       bert_model_name=bert_model_name,
                                       bert_initial_model=bert_initial_model)
        super().__init__(module, input_dtype=torch.long, optimizer=AdamW(module.parameters(), lr=lr), verbose=verbose)
