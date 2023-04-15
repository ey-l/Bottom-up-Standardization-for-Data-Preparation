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
sub_df = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_df.info()
test_df.info()
(train_df.shape, test_df.shape)
plt.figure(figsize=[12, 4])
plt.subplot(1, 2, 1)
(100 * train_df.isna().sum() / train_df.shape[0]).plot.bar(title='Missing Values Ratio in Train Data', ylabel='Rario(%)', xlabel='Column Name')
plt.subplot(1, 2, 2)
(100 * test_df.isna().sum() / test_df.shape[0]).plot.bar(title='Missing Value Ratio in Test Data', ylabel='Rario(%)', xlabel='Column Name')
train_df.loc[train_df['text'].duplicated(keep=False)].sort_values(by='text')
post_df = pd.DataFrame(train_df.loc[train_df['text'].duplicated(keep=False)].groupby('text')['target'].mean())
train_df['target'].value_counts().plot.bar(xlabel='Colum Name', ylabel='Count', title='Distribution of Label')
import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub('http[s]?\\:\\/\\/\\S+', ' ', text)
    text = re.sub('[ \t\n]+', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text.strip().lower()
for df in (train_df, test_df):
    df.text = df.text.apply(lambda x: clean_text(x))
train_df['text'].apply(lambda x: x.split(' ')).str.len().plot.hist(title='Text length', xlabel='Number of Words', ylabel='Count')
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_df['text'])
test_vectors = vectorizer.transform(test_df['text'])

def get_validation_metrics(clf, df, vector_):
    scores = model_selection.cross_validate(clf, vector_, df['target'], cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'], return_train_score=True)
    return pd.DataFrame(scores)
clf_ridge = RidgeClassifier()
get_validation_metrics(clf_ridge, train_df, train_vectors).mean()
clf_lr = LogisticRegression()
get_validation_metrics(clf_lr, train_df, train_vectors).mean()