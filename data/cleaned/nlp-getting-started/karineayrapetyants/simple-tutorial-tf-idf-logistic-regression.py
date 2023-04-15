import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
train.head()
train.isnull().sum(axis=0)
len(train)
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
test.head()
(X, Y) = (train['text'], train['target'])
X_test = test['text']
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_vect_train = vectorizer.fit_transform(X)
X_vect_test = vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression