import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1[_input1['target'] == 0]['text'].values[1]
_input1[_input1['target'] == 1]['text'].values[1]
_input1.describe()
_input1.info()
count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(_input1['text'][:5])
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(_input1['text'])
test_vectors = count_vectorizer.transform(_input0['text'])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, _input1['target'], cv=3, scoring='f1')
scores