import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import csv
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.columns
_input1['target'].value_counts()
(_input1.shape, _input0.shape)
train0 = _input1[_input1['target'] == 0]
train1 = _input1[_input1['target'] == 1]
(train0.shape, train1.shape)
_input1['text'] = _input1['text'].replace({'#(\\w+)': ''}, inplace=False, regex=True)
_input1['text'] = _input1['text'].replace({'@(\\w+)': ''}, inplace=False, regex=True)
_input1['text'] = _input1['text'].astype(str).replace({'http\\S+': ''}, inplace=False, regex=True)
_input1['text'] = _input1['text'].str.lower()
_input0['text'] = _input0['text'].replace({'#(\\w+)': ''}, inplace=False, regex=True)
_input0['text'] = _input0['text'].replace({'@(\\w+)': ''}, inplace=False, regex=True)
_input0['text'] = _input0['text'].astype(str).replace({'http\\S+': ''}, inplace=False, regex=True)
_input0['text'] = _input0['text'].str.lower()
from nltk.corpus import stopwords
stop = stopwords.words('english')
_input1['text'] = _input1['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
_input0['text'] = _input0['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
_input1.head()
_input0.head()
y = _input1['target']
x = _input1['text']
x.shape
corpus = []
for i in range(x.shape[0]):
    corpus.append(x.iloc[i])
vectorizer1 = TfidfVectorizer(max_features=1000)
X1 = vectorizer1.fit_transform(x)
feature_names1 = vectorizer1.get_feature_names()
denselist1 = X1.todense().tolist()
_input1 = pd.DataFrame(denselist1, columns=feature_names1)
(X_temp, X_test, y_temp, y_test) = train_test_split(_input1, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
accuracy = {'TF-IDF': []}
regressor_LR_tf = LogisticRegression(C=1.0, penalty='l2', solver='newton-cg')