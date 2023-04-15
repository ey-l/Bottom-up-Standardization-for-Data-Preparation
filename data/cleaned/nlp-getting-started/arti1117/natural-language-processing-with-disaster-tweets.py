import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_df.head()
train_df[train_df['target'] == 0]['text'].values[1]
train_df[train_df['target'] == 1]['text'].values[1]
train_df.describe()
train_df.info()
count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(train_df['text'][:5])
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df['text'])
test_vectors = count_vectorizer.transform(test_df['text'])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df['target'], cv=3, scoring='f1')
scores