import sys
import numpy as np
import torch
from octoml.bert.bert_for_classification import BertForClassification
from octoml.bert.constants import BERT_BASE_UNCASED_MODEL_NAME
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

EPOCHS = 5
BATCH_SIZE = 4

class BertClassifier(BaseEstimator, ClassifierMixin):
    """ A sklearn like class for text classification based on BERT.
    This class uses BERT and a linear layer on top of it to perform binary/multi-class classification. The loss function
    is determined automatically by the shape of the target variable.
    """
    def __init__(self, bert_model_name=BERT_BASE_UNCASED_MODEL_NAME,
                 verbose=True,
                 bert_initial_model=None,
                 lr=1e-3):
        """
        bert_model:  The name of the BERT model to use (default is 'bert-base-multilingual-cased'). Accepts one if the values from here: https://github.com/huggingface/pytorch-pretrained-BERT/blob/98dc30b21e3df6528d0dd17f0910ffea12bc0f33/pytorch_pretrained_bert/modeling.py#L36
        verbose:  If True (by default, outputs the training progress (epochs, steps and loss)
        bert_initial_model: A pytroch implementation for BERT, will be used as base model instead of the one provided in "bert_model"
        """
        self.verbose = verbose
        self.model_name = bert_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_initial_model = bert_initial_model
        self.model = None
        self.lr = lr
        self.loss_func = CrossEntropyLoss()

    def fit(self, X, y, epochs=EPOCHS, output_dim=None):
        """
        Fine-tunes BERT and trains the final linear layer

        X: A numpy array of vectorized text using BertVectorizer
        y: A numpy array of the target variable.
        epochs:  Number of epochs to fine-tune, default is 5
        output_dim: The output dimensions of the final linear layer, will be computed automatically from the target variable if None
        return: self
        """
        if output_dim:
            self.output_dim = output_dim
        else:
            if len(y.shape) > 2:
                raise Exception('Target shape must be of dim 1 or 2')
            elif len(y.shape) > 1:
                self.output_dim = y.shape[1]
            else:
                self.output_dim = int(max(y) + 1)

        self.model = BertForClassification(output_size=self.output_dim,
                                           bert_model_name=self.model_name,
                                           bert_initial_model=self.bert_initial_model)

        inputs = torch.tensor(X)
        tags = torch.tensor(y, dtype=torch.long)
        masks = torch.tensor(X > 0, dtype=torch.long)

        data = TensorDataset(inputs, masks, tags)
        sampler = RandomSampler(data)
        loader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)

        if str(self.device) != 'cpu':
            self.model.cuda()

        self.fine_tune_model(input_size=inputs.shape[0],
                            data_loader=loader,
                            epochs=epochs)

        return self

    def predict(self, X):
        """
        Uses the model to predict the target variable

        X: A numpy array of vectorized text using BertVectorizer
        return: A numpy array with predictions - class id for each instance
        """
        logits = self.decision_function(X)
        return np.argmax(logits, axis=1)

    def decision_function(self, X):
        """
         Uses the model to predict the target variable and returns the final layer output without activation

        X: A numpy array of vectorized text using BertVectorizer
        return: A numpy array with raw logtis. One vector for each instance
        """
        if self.model is None:
            raise Exception("BertClassifier is not fitted yet")

        self.model.eval()

        inputs = torch.tensor(X)
        masks = torch.tensor(X > 0, dtype=torch.long)

        data = TensorDataset(inputs, masks)
        loader = DataLoader(data, batch_size=BATCH_SIZE)

        all_logits = []
        for step_num, (token_ids, input_mas) in enumerate(loader):
            token_ids, input_mask = token_ids.to(self.device), input_mas.to(self.device)
            with torch.no_grad():
                logits = self.model(token_ids, attention_mask=input_mask)
            all_logits.append(logits.cpu().detach().numpy())
        return np.vstack(all_logits)

    def fine_tune_model(self, input_size, data_loader, epochs):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        num_of_steps = input_size // BATCH_SIZE + (input_size % BATCH_SIZE > 0)
        for epoch_num in range(epochs):
            if self.verbose:
                sys.stdout.write("Epoch {0}/{1}:\n".format(epoch_num+1, epochs))
                sys.stdout.flush()

            self.train_epoch(num_of_steps, optimizer, data_loader)

    def train_epoch(self, num_of_steps, optimizer, train_loader):
        self.model.train()
        epoch_loss = 0.0
        for step_num, (token_ids, input_mask, tags) in enumerate(train_loader):
            token_ids, input_mask, tags = token_ids.to(self.device), input_mask.to(self.device), tags.to(self.device)

            self.model.zero_grad()

            logits = self.model(token_ids, attention_mask=input_mask)

            batch_loss = self.loss_func(logits.view(-1, self.output_dim), tags.view(-1))
            epoch_loss += batch_loss.item()

            batch_loss.backward()
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
            optimizer.step()

            if self.verbose:
                sys.stdout.write("\r" + "{0}/{1} batch loss: {2} ".format(step_num, num_of_steps, batch_loss.item()))
                sys.stdout.flush()

        if self.verbose:
            sys.stdout.write("avg loss: {0}\n".format(epoch_loss / num_of_steps))
            sys.stdout.flush()
