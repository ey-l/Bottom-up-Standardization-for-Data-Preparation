import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.info()
_input1
_input1['text length'] = _input1['text'].apply(len)
_input1.head()
g = sns.FacetGrid(_input1, col='target')
g.map(sns.histplot, 'text length', bins=35)
_input1['location'].isnull().sum()
_input1['keyword'].isnull().sum()
_input1['keyword'].unique()
sns.countplot(data=_input1, x='target')
_input1['keyword'] = _input1['keyword'].str.replace('%20', ' ')
_input1['keyword'].nunique()
_input1[_input1['target'] == 0]
_input1[_input1['target'] == 1]
_input1['keyword'] = _input1['keyword'].fillna(_input1['keyword'].mode()[0])
_input1['location'] = _input1['location'].fillna(_input1['location'].mode()[0])
plt.figure(figsize=(12, 12))
sns.heatmap(_input1.isna())
_input1['new text'] = _input1['text'] + ' ' + _input1['location'] + ' ' + _input1['keyword']
_input1['new text'] = _input1['new text'].str.replace('#', ' ')
_input1.head()
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0
_input0['location'] = _input0['location'].fillna(_input0['location'].mode()[0])
_input0['keyword'] = _input0['keyword'].fillna(_input0['keyword'].mode()[0])
_input0['keyword'] = _input0['keyword'].str.replace('%20', ' ')
_input0['keyword'].nunique()
_input0.isnull().sum()
plt.figure(figsize=(12, 12))
sns.heatmap(_input0.isna())
_input0['new text'] = _input0['text'] + ' ' + _input0['location'] + ' ' + _input0['keyword']
_input0['new text'] = _input0['new text'].str.replace('#', ' ')
_input0.head()
_input1.head()
_input0.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
X_train = _input1['new text']
y_train = _input1['target']
X_test = _input0['new text']
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
nb = MultinomialNB()