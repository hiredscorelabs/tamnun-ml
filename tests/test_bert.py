import numpy as np
from tamnun.bert import BertClassifier, BertVectorizer

def test_multiclass_predictions_shapes():
    X = np.array([[101, 429, 876],
                  [101, 429, 876],
                  [101, 429, 876]])
    y = np.array([1, 2, 1])

    clf = BertClassifier(num_of_classes=3).fit(X, y, epochs=1)
    predicted = clf.predict(X)

    assert predicted.shape == (3,)


def test_binary_predictions_shapes():
    X = np.array([[101, 429, 876],
                  [101, 429, 876],
                  [101, 429, 876]])
    y = np.array([1, 0, 1])

    clf = BertClassifier(num_of_classes=2).fit(X, y, epochs=1)
    predicted = clf.predict(X)

    assert predicted.shape == (3,)


def test_vectorizer_shapes():
    X = ['Hi, my name is Dima']
    assert BertVectorizer().fit_transform(X).shape == (1, 8)
    assert BertVectorizer(max_len=300).fit_transform(X).shape == (1, 300)

    X = ['Hi, my name is Dima'*500]
    assert BertVectorizer(do_truncate=True).fit_transform(X).shape == (1, 512)
