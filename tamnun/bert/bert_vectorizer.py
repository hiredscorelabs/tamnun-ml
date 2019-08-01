import numpy as np
from sklearn.base import TransformerMixin
from pytorch_transformers import BertTokenizer
from tamnun.bert.constants import BERT_BASE_UNCASED_MODEL_NAME


CLS_TOKEN = '[CLS]'
PAD_TOKEN = '[PAD]'
MAX_SEQUENCE_LEN = 512


class BertVectorizer(TransformerMixin):
    """
    A sklearn like class for text vectorization for BertClassifier.
    """
    def __init__(self, bert_model=BERT_BASE_UNCASED_MODEL_NAME, max_len='auto', do_lower_case=True, do_truncate=False):
        """
        bert_model: The name of the BERT model to use (default is 'bert-base-multilingual-cased'). Accepts one if the values from here: https://github.com/huggingface/pytorch-pretrained-BERT/blob/98dc30b21e3df6528d0dd17f0910ffea12bc0f33/pytorch_pretrained_bert/modeling.py#L36
        max_len: Max possible length of the sequence to use (default is 'auto'). Will be calculated if not provided. The max possible lenght for BERT is 512.
        do_lower_case: Used in the tokenizing stage. Should be corresponding to the BERT model (cased/uncased)
        do_truncate: If of of the examples in the data is longer than `max_len`, exception will be thrown unless `do_trunncate=True` and then the data will be truncated.
        """
        self.do_truncate = do_truncate
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

        if max_len != 'auto' and (not isinstance(max_len, int) or max_len < 0 or max_len > MAX_SEQUENCE_LEN):
                raise Exception("max_len should be a number between 0 and {0} or 'auto'".format(MAX_SEQUENCE_LEN))

        self.max_len = max_len

    def fit(self, X, y=None, **fit_params):
        """
        Calculates the max_len if it's 'auto' if `max_len` is provided, there's no need in this method
        X: A list of strings
        return: self
        """
        tokenized = list(map(self.tokenize_instance, X))
        data_max_len = len(max(tokenized, key=len))
        if self.max_len == 'auto':
            self.max_len = data_max_len

        if self.max_len > MAX_SEQUENCE_LEN and not self.do_truncate:
            raise Exception('Max sequence length is {0}, use do_truncate=True if you want to truncate'.format(self.max_len))
        elif self.max_len > MAX_SEQUENCE_LEN:
            self.max_len = MAX_SEQUENCE_LEN

        return self

    def transform(self, X, y=None):
        """
        Tokenizes the text, converts it to tokenizer id and pads/truncates to the desired `max_len`

        X:  A list of strings
        return: A numpy array of size (N, max_len) when each instance is a vector of token ids padded/truncated
        """
        if self.max_len == 'auto':
            raise Exception("BertVectorizer is not fitted yet (max_len=auto)")
        tokens = map(self.tokenize_instance, X)
        tokens = map(self.truncate_and_pad, tokens)
        tokens_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))
        return np.array(tokens_ids)

    def tokenize_instance(self, x):
        return [CLS_TOKEN] + self.tokenizer.tokenize(x)

    def truncate_and_pad(self, x):
        return x[:self.max_len] + [PAD_TOKEN]*(self.max_len - len(x))
