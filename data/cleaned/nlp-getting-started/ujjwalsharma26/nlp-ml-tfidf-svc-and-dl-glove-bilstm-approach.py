import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_df.head(5)
train_df.isnull().sum()
test_df.isnull().sum()
train_df = train_df.fillna(' ')
train_df.isnull().sum()
test_df = test_df.fillna(' ')
test_df.isnull().sum()
train_df['text'] = train_df['keyword'] + ' ' + train_df['location'] + ' ' + train_df['text']
test_df['text'] = test_df['keyword'] + ' ' + test_df['location'] + ' ' + test_df['text']
train_df = train_df.drop('keyword', axis=1)
train_df = train_df.drop('location', axis=1)
test_df = test_df.drop('keyword', axis=1)
test_df = test_df.drop('location', axis=1)
print(train_df['text'][5])
import re

def text_normalize(sen):
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    sentence = re.sub('\\s+[a-zA-Z]\\s+', ' ', sentence)
    sentence = re.sub('\\s+', ' ', sentence)
    return sentence.lower()
X = []
for sen in list(train_df['text']):
    X.append(text_normalize(sen))
train_df['text'] = X
X = []
for sen in list(test_df['text']):
    X.append(text_normalize(sen))
test_df['text'] = X
train_df.head()
print(train_df['text'][5])
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(train_df['text'], train_df['target'], test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC(loss='hinge', fit_intercept=False))])