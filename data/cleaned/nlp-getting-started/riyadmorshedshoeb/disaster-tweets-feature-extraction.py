import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.isna().sum()
test.isna().sum()
train.fillna('missing', inplace=True)
test.fillna('missing', inplace=True)
train.iloc[0]
(X_train, X_test, y_train, y_test) = train_test_split(train[['keyword', 'location', 'text']], train['target'], test_size=0.2, random_state=42)
bow_key = CountVectorizer()
X_train_key = bow_key.fit_transform(X_train['keyword']).toarray()
X_test_key = bow_key.transform(X_test['keyword']).toarray()
test_key = bow_key.transform(test['keyword']).toarray()
bow_loc = CountVectorizer()
X_train_loc = bow_loc.fit_transform(X_train['location']).toarray()
X_test_loc = bow_loc.transform(X_test['location']).toarray()
test_loc = bow_loc.transform(test['location']).toarray()
tfidf = TfidfVectorizer(ngram_range=(1, 2))
X_train_text = tfidf.fit_transform(X_train['text']).toarray()
X_test_text = tfidf.transform(X_test['text']).toarray()
test_text = tfidf.transform(test['text']).toarray()
len(tfidf.get_feature_names())
X_train_vec = np.concatenate((X_train_key, X_train_loc, X_train_text), axis=1)
X_test_vec = np.concatenate((X_test_key, X_test_loc, X_test_text), axis=1)
test_vec = np.concatenate((test_key, test_loc, test_text), axis=1)
X_train_vec.shape