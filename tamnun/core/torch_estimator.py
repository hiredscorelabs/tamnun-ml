import sys
import torch
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


TASK_TYPE_TO_LOSS_FUNCTION = {
    'classification': CrossEntropyLoss(),
    'regression': MSELoss(),
    'multitag': BCEWithLogitsLoss()
}

class TorchEstimator(object):
    def __init__(self, torch_module, optimizer=None, task_type='classification', input_dtype=torch.float, verbose=True):
        self.model = torch_module
        self.verbose = verbose
        self.input_dtype = input_dtype
        self.loss_function = get_loss_function_for_task(task_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_type = task_type
        self.optimizer = optimizer or Adam(self.model.parameters())

    def fit(self, X, y, output_dim=None, batch_size=4, epochs=5):
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
                self.output_dim = 1


        inputs = torch.tensor(X, dtype=self.input_dtype)
        tags = torch.tensor(y, dtype=torch.float if self.task_type == 'regression' else torch.long)

        data = TensorDataset(inputs, tags)
        sampler = RandomSampler(data)
        loader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        if torch.cuda.is_available(): self.model.cuda()

        self.train_model(input_size=inputs.shape[0], data_loader=loader, epochs=epochs, batch_size=batch_size)

        return self

    def train_model(self, input_size, data_loader, epochs, batch_size):
        num_of_steps = input_size // batch_size + (input_size % batch_size > 0)
        for epoch_num in range(epochs):
            if self.verbose:
                sys.stdout.write("Epoch {0}/{1}:\n".format(epoch_num+1, epochs))
                sys.stdout.flush()

            self.train_epoch(num_of_steps, data_loader)

    def train_epoch(self, num_of_steps, train_loader):
        self.model.train()
        epoch_loss = 0.0
        for step_num, (input, tags) in enumerate(train_loader):
            input, tags = input.to(self.device), tags.to(self.device)

            self.model.zero_grad()

            logits = self.model(input)

            if self.task_type != 'classification':
                tags = tags.view(-1, self.output_dim)

            batch_loss = self.loss_function(logits, tags)
            epoch_loss += batch_loss.item()

            batch_loss.backward()
            self.optimizer.step()

            if self.verbose:
                sys.stdout.write("\r" + "{0}/{1} batch loss: {2} ".format(step_num, num_of_steps, batch_loss.item()))
                sys.stdout.flush()

        if self.verbose:
            sys.stdout.write("avg loss: {0}\n".format(epoch_loss / num_of_steps))
            sys.stdout.flush()


    def predict(self, X, batch_size=4):
        """
        Uses the model to predict the target variable

        X: A numpy array of vectorized text using BertVectorizer
        return: A numpy array with predictions - class id for each instance
        """
        logits = self.decision_function(X, batch_size=batch_size)
        if self.task_type == 'classification':
            return np.argmax(logits, axis=1)
        elif self.task_type == 'regression':
            if self.output_dim == 1:
                logits = logits.reshape(-1,)
            return logits
        elif self.task_type == 'multitag':
            return logits > 0.5

    def decision_function(self, X, batch_size=4):
        """
         Uses the model to predict the target variable and returns the final layer output without activation

        X: A numpy array of vectorized text using BertVectorizer
        return: A numpy array with raw logtis. One vector for each instance
        """
        if self.model is None:
            raise Exception("BertClassifier is not fitted yet")

        self.model.eval()

        inputs = torch.tensor(X, dtype=self.input_dtype)

        data = TensorDataset(inputs)
        loader = DataLoader(data, batch_size=batch_size)

        all_logits = []
        for step_num, (input, ) in enumerate(loader):
            input = input.to(self.device)

            with torch.no_grad():
                logits = self.model(input)
            all_logits.append(logits.cpu().detach().numpy())
        return np.vstack(all_logits)


def get_loss_function_for_task(task_type):
    if task_type not in TASK_TYPE_TO_LOSS_FUNCTION:
        raise Exception("Unknown task type '{task_type}, the options are {options}".format(task_type=task_type,
                                                                                           options=list(TASK_TYPE_TO_LOSS_FUNCTION.keys())))

    return TASK_TYPE_TO_LOSS_FUNCTION[task_type]