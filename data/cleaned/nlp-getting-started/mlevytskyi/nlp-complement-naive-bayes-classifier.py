import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from pprint import pprint
from sklearn.metrics import auc, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
import gc
import string
import re
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
print('train shape:', train.shape)
print('test shape:', test.shape)
train.head()
test.head()
train['keyword'].fillna('', inplace=True)
train['location'].fillna('', inplace=True)
test['keyword'].fillna('', inplace=True)
test['location'].fillna('', inplace=True)
train['final_text'] = train['keyword'] + ' ' + train['text'] + ' ' + train['location']
test['final_text'] = test['keyword'] + ' ' + test['text'] + ' ' + test['location']

def remove_URL(text):
    url = re.compile('https?://\\S+|www\\.\\S+')
    return url.sub('', text)

def remove_html(text):
    html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)
train['final_text'] = train['final_text'].apply(lambda x: remove_URL(x))
train['final_text'] = train['final_text'].apply(lambda x: remove_html(x))
train['final_text'] = train['final_text'].apply(lambda x: remove_punct(x))
test['final_text'] = test['final_text'].apply(lambda x: remove_URL(x))
test['final_text'] = test['final_text'].apply(lambda x: remove_html(x))
test['final_text'] = test['final_text'].apply(lambda x: remove_punct(x))
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train['final_text'])
test_vectors = count_vectorizer.transform(test['final_text'])
y = train['target']
del train, test
gc.collect()
(X_train, X_test, y_train, y_test) = train_test_split(train_vectors.toarray(), y, test_size=0.33, random_state=42)
del train_vectors
gc.collect()
clf = ComplementNB()