from torch import nn
import numpy as np
from torch.optim import SGD
from tamnun.core.torch_estimator import TorchEstimator


def test_classifier_default():
    module = nn.Linear(10, 2)
    clf = TorchEstimator(module)
    X = np.array([list(range(10)),
                  list(range(10, 20))])
    y = np.array([0, 1])
    predicted = clf.fit(X, y, epochs=1000).predict(X)

    assert (predicted == y).all()


def test_multiclass():
    module = nn.Linear(10, 3)
    clf = TorchEstimator(module)
    X = np.array([list(range(10)),
                  list(range(100, 110)),
                  list(range(10)),
                  list(range(10, 20))])
    y = np.array([0, 2, 0, 1])
    predicted = clf.fit(X, y, epochs=2000).predict(X)

    assert (predicted == y).all()


def test_regression_custom_optimizer():
    module = nn.Linear(1, 1, bias=False)
    reg = TorchEstimator(module, task_type='regression', optimizer=SGD(module.parameters(), lr=1e-4))
    X = np.array([[1], [10]])
    y = np.array([10, 100])
    predicted = reg.fit(X, y, epochs=1000, batch_size=2).predict(X)

    assert (np.round(predicted) == y).all()


def test_multi_regression():
    module = nn.Linear(1, 2, bias=False)
    reg = TorchEstimator(module, task_type='regression', optimizer=SGD(module.parameters(), lr=1e-4))
    X = np.array([[1], [10]])
    y = np.array([[10, 50], [100, 500]])
    predicted = reg.fit(X, y, epochs=2000, batch_size=2).predict(X)

    assert (np.round(predicted) == y).all()
