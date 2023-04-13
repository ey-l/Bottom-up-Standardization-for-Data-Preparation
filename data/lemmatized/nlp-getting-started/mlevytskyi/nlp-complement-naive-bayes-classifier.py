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
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
print('train shape:', _input1.shape)
print('test shape:', _input0.shape)
_input1.head()
_input0.head()
_input1['keyword'] = _input1['keyword'].fillna('', inplace=False)
_input1['location'] = _input1['location'].fillna('', inplace=False)
_input0['keyword'] = _input0['keyword'].fillna('', inplace=False)
_input0['location'] = _input0['location'].fillna('', inplace=False)
_input1['final_text'] = _input1['keyword'] + ' ' + _input1['text'] + ' ' + _input1['location']
_input0['final_text'] = _input0['keyword'] + ' ' + _input0['text'] + ' ' + _input0['location']

def remove_URL(text):
    url = re.compile('https?://\\S+|www\\.\\S+')
    return url.sub('', text)

def remove_html(text):
    html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)
_input1['final_text'] = _input1['final_text'].apply(lambda x: remove_URL(x))
_input1['final_text'] = _input1['final_text'].apply(lambda x: remove_html(x))
_input1['final_text'] = _input1['final_text'].apply(lambda x: remove_punct(x))
_input0['final_text'] = _input0['final_text'].apply(lambda x: remove_URL(x))
_input0['final_text'] = _input0['final_text'].apply(lambda x: remove_html(x))
_input0['final_text'] = _input0['final_text'].apply(lambda x: remove_punct(x))
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(_input1['final_text'])
test_vectors = count_vectorizer.transform(_input0['final_text'])
y = _input1['target']
del _input1, _input0
gc.collect()
(X_train, X_test, y_train, y_test) = train_test_split(train_vectors.toarray(), y, test_size=0.33, random_state=42)
del train_vectors
gc.collect()
clf = ComplementNB()