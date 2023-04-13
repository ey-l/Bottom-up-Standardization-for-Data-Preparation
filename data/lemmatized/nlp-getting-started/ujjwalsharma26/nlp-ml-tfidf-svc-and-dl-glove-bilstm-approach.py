import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head(5)
_input1.isnull().sum()
_input0.isnull().sum()
_input1 = _input1.fillna(' ')
_input1.isnull().sum()
_input0 = _input0.fillna(' ')
_input0.isnull().sum()
_input1['text'] = _input1['keyword'] + ' ' + _input1['location'] + ' ' + _input1['text']
_input0['text'] = _input0['keyword'] + ' ' + _input0['location'] + ' ' + _input0['text']
_input1 = _input1.drop('keyword', axis=1)
_input1 = _input1.drop('location', axis=1)
_input0 = _input0.drop('keyword', axis=1)
_input0 = _input0.drop('location', axis=1)
print(_input1['text'][5])
import re

def text_normalize(sen):
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    sentence = re.sub('\\s+[a-zA-Z]\\s+', ' ', sentence)
    sentence = re.sub('\\s+', ' ', sentence)
    return sentence.lower()
X = []
for sen in list(_input1['text']):
    X.append(text_normalize(sen))
_input1['text'] = X
X = []
for sen in list(_input0['text']):
    X.append(text_normalize(sen))
_input0['text'] = X
_input1.head()
print(_input1['text'][5])
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(_input1['text'], _input1['target'], test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC(loss='hinge', fit_intercept=False))])