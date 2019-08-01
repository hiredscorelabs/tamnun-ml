import numpy as np
from torchnlp.datasets import imdb_dataset # run pip install pytorch-nlp if you dont have this
from tamnun.bert import BertClassifier, BertVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Getting data
train_data, test_data = imdb_dataset(train=True, test=True)

# Each instance in the dataset is a dict with `text` and `sentiment`, we extract those fields and create two variables
train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), train_data)))
test_texts, test_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), test_data)))

# Convert label to boolean
train_y = np.array(train_labels) == 'pos'
test_y = np.array(test_labels) == 'pos'

print('Train size:', len(train_texts))
print('Test size:', len(test_texts))

# Create a pipeline with the vectorizer and the classifier and then fit it on the raw data
print('Fine-tuning BERT...')
clf = make_pipeline(BertVectorizer(do_truncate=True),
                    BertClassifier(num_of_classes=2, lr=1e-5)).fit(train_texts, train_y)

predicted = clf.predict(test_texts)

print(classification_report(predicted, test_y))

