from torch import nn
from torch.optim import Adam
import torchvision.datasets as datasets # pip install torch vision if you dont have this
from sklearn.metrics import classification_report
from tamnun.core import TorchEstimator

# Load data using torch vision
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# Get the data tensors for train and test
train_X = mnist_trainset.data.reshape(-1, 28*28)
train_y = mnist_trainset.targets

test_X = mnist_testset.data.reshape(-1, 28*28)
test_y = mnist_testset.targets

# Create simple linear classifier with 28x28 (the image size) as input and 10 classes as output
module = nn.Linear(28*28, 10)

# Create the tamnun estimator
clf = TorchEstimator(module, optimizer=Adam(module.parameters(), lr=1e-4))

# fit().predict()!
clf.fit(train_X, train_y, epochs=10, batch_size=32)
predicted = clf.predict(test_X)
print(classification_report(test_y, predicted))
