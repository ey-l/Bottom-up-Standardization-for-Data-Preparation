import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import feature_extraction, model_selection, preprocessing
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk import ngrams
sns.set_theme()
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.info()
_input0.info()
(_input1.shape, _input0.shape)
plt.figure(figsize=[12, 4])
plt.subplot(1, 2, 1)
(100 * _input1.isna().sum() / _input1.shape[0]).plot.bar(title='Missing Values Ratio in Train Data', ylabel='Rario(%)', xlabel='Column Name')
plt.subplot(1, 2, 2)
(100 * _input0.isna().sum() / _input0.shape[0]).plot.bar(title='Missing Value Ratio in Test Data', ylabel='Rario(%)', xlabel='Column Name')
_input1.loc[_input1['text'].duplicated(keep=False)].sort_values(by='text')
post_df = pd.DataFrame(_input1.loc[_input1['text'].duplicated(keep=False)].groupby('text')['target'].mean())
_input1['target'].value_counts().plot.bar(xlabel='Colum Name', ylabel='Count', title='Distribution of Label')
import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub('http[s]?\\:\\/\\/\\S+', ' ', text)
    text = re.sub('[ \t\n]+', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text.strip().lower()
for df in (_input1, _input0):
    df.text = df.text.apply(lambda x: clean_text(x))
_input1['text'].apply(lambda x: x.split(' ')).str.len().plot.hist(title='Text length', xlabel='Number of Words', ylabel='Count')
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(_input1['text'])
test_vectors = vectorizer.transform(_input0['text'])

def get_validation_metrics(clf, df, vector_):
    scores = model_selection.cross_validate(clf, vector_, df['target'], cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'], return_train_score=True)
    return pd.DataFrame(scores)
clf_ridge = RidgeClassifier()
get_validation_metrics(clf_ridge, _input1, train_vectors).mean()
clf_lr = LogisticRegression()
get_validation_metrics(clf_lr, _input1, train_vectors).mean()