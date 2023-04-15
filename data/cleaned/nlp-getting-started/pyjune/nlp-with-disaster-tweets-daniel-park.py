import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import re
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
train[train['target'] == 0]['text'].values[1]
train[train['target'] == 1]['text'].values[1]
train.info()
train.isnull().sum()
count_vectorizer = feature_extraction.text.CountVectorizer()
ex_train_vectors = count_vectorizer.fit_transform(train['text'][0:5])
print(ex_train_vectors[0].todense().shape)
train_vectors = count_vectorizer.fit_transform(train['text'])
test_vectors = count_vectorizer.transform(test['text'])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train['target'], cv=3, scoring='f1')
scores