import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score

train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.shape
test.shape
train.head()
train.dtypes
train['target'].sum() / train['target'].count() * 100
nlp = spacy.load('en_core_web_sm')

def text_preprocessing(string):
    doc = nlp(string)
    doc2 = ' '.join([str(token) for token in doc if str(token).isalpha()])
    doc3 = nlp(doc2)
    return ' '.join([token.lemma_ for token in doc3])
train['pre-processed text'] = train['text'].apply(text_preprocessing)
test['pre-processed text'] = test['text'].apply(text_preprocessing)
train[['text', 'pre-processed text']].head()
model = MultinomialNB()
vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X = np.array(train['pre-processed text'])
y = np.array(train['target'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, stratify=y, random_state=12)
kf = KFold(n_splits=10, shuffle=True, random_state=451)
scores = []
for (train_index, test_index) in kf.split(X):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    bow_train = vectorizer.fit_transform(X_train)
    bow_test = vectorizer.transform(X_test)