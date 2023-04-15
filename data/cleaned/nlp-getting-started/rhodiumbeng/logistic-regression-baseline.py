import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
submission = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
train_df.info()
train_df.head()
train_df['target'].value_counts()
X = train_df['text']
y = train_df['target']
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=123)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
print(y_train.value_counts(), '\n', y_val.value_counts())
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = CountVectorizer(lowercase=True, stop_words='english', token_pattern='(?u)\\b\\w+\\b|\\,|\\.|\\;|\\:')
vect
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm
X_val_dtm = vect.transform(X_val)
X_val_dtm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=8)
logreg
from sklearn.model_selection import GridSearchCV
grid_values = {'C': [0.01, 0.1, 1.0, 3.0, 5.0]}
grid_logreg = GridSearchCV(logreg, param_grid=grid_values, scoring='neg_log_loss', cv=5)