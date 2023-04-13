import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.isnull().sum(axis=0)
len(_input1)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0.head()
(X, Y) = (_input1['text'], _input1['target'])
X_test = _input0['text']
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_vect_train = vectorizer.fit_transform(X)
X_vect_test = vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression