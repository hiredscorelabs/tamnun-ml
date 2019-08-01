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
    """
    Torch module wrapper with the sklearn interface for training and predictions.

    torch_module: a model of type torch.nn.Module
    optimizer (optional): optimizer from toch.optim, if none, Adam with default params is used.
    task_type (optional): one of 'classification' (default), 'regression', 'multitag'. Sets the shapes, the loss func and more.
    input_dype (optional): the tensor type should be created for the input, default is float
    verbose (optional): if True (default) print training progress (epochs and loss)
    """
    def __init__(self, torch_module, optimizer=None, task_type='classification', input_dtype=torch.float, verbose=True):
        self.output_dim = 1
        self.model = torch_module
        self.verbose = verbose
        self.input_dtype = input_dtype
        self.loss_function = get_loss_function_for_task(task_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_type = task_type
        self.optimizer = optimizer or Adam(self.model.parameters())

    def fit(self, X, y, output_dim=None, batch_size=4, epochs=5):
        """
        fits the model

        X: A numpy array or torch tensor for the input.
        y: A numpy array or torch tensor for the target variable.
        batch_size (optional): the batch size to use during training, default=4
        epochs (optional):  Number of epochs to train, default is 5
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

            output = self.model(input)

            if self.task_type != 'classification':
                tags = tags.view(-1, self.output_dim)

            batch_loss = self.loss_function(output, tags)
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

        X: A numpy array or torch tensor for the input.
        return: A numpy array with predictions
        """
        output = self.decision_function(X, batch_size=batch_size)
        if self.task_type == 'classification':
            return np.argmax(output, axis=1)
        elif self.task_type == 'regression':
            if self.output_dim == 1:
                output = output.reshape(-1,)
            return output
        elif self.task_type == 'multitag':
            return output > 0.5

    def decision_function(self, X, batch_size=4):
        """
         Uses the model to predict the target variable and returns the final layer output without activation

        X: A numpy array or torch tensor for the input.
        return: A numpy array with raw output of the final layer. One vector for each instance
        """
        if self.model is None:
            raise Exception("BertClassifier is not fitted yet")

        self.model.eval()

        inputs = torch.tensor(X, dtype=self.input_dtype)

        data = TensorDataset(inputs)
        loader = DataLoader(data, batch_size=batch_size)

        all_output = []
        for step_num, (input, ) in enumerate(loader):
            input = input.to(self.device)

            with torch.no_grad():
                output = self.model(input)
            all_output.append(output.cpu().detach().numpy())
        return np.vstack(all_output)


def get_loss_function_for_task(task_type):
    if task_type not in TASK_TYPE_TO_LOSS_FUNCTION:
        raise Exception("Unknown task type '{task_type}, the options are {options}".format(task_type=task_type,
                                                                                           options=list(TASK_TYPE_TO_LOSS_FUNCTION.keys())))

    return TASK_TYPE_TO_LOSS_FUNCTION[task_type]