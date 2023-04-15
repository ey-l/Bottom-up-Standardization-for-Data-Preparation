import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
(train_df.shape, test_df.shape)
train_df.info()
train_df.describe()
train_df.head()
test_df.head()
train_df[train_df['target'] == 0]['text'].values[0]
train_df[train_df['target'] == 1]['text'].values[0]
count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(train_df['text'][0:5])
'we use .todense() here because these vectors are "sparse"\n(only non-zero elements are kept to save space)\n'
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df['text'])
"Note that we're NOT using .fit_transform() here. \nUsing just .transform() makes sure that the tokens in the train vectors\nare the only ones mapped to the test vectors - \ni.e. that the train and test vectors use the same set of tokens. "
test_vectors = count_vectorizer.transform(test_df['text'])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df['target'], cv=3, scoring='f1')
scores