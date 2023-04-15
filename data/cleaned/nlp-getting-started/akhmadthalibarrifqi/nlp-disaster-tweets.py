import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
train_df.info()
train_df
train_df['text length'] = train_df['text'].apply(len)
train_df.head()
g = sns.FacetGrid(train_df, col='target')
g.map(sns.histplot, 'text length', bins=35)
train_df['location'].isnull().sum()
train_df['keyword'].isnull().sum()
train_df['keyword'].unique()
sns.countplot(data=train_df, x='target')
train_df['keyword'] = train_df['keyword'].str.replace('%20', ' ')
train_df['keyword'].nunique()
train_df[train_df['target'] == 0]
train_df[train_df['target'] == 1]
train_df['keyword'] = train_df['keyword'].fillna(train_df['keyword'].mode()[0])
train_df['location'] = train_df['location'].fillna(train_df['location'].mode()[0])
plt.figure(figsize=(12, 12))
sns.heatmap(train_df.isna())
train_df['new text'] = train_df['text'] + ' ' + train_df['location'] + ' ' + train_df['keyword']
train_df['new text'] = train_df['new text'].str.replace('#', ' ')
train_df.head()
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
test_df
test_df['location'] = test_df['location'].fillna(test_df['location'].mode()[0])
test_df['keyword'] = test_df['keyword'].fillna(test_df['keyword'].mode()[0])
test_df['keyword'] = test_df['keyword'].str.replace('%20', ' ')
test_df['keyword'].nunique()
test_df.isnull().sum()
plt.figure(figsize=(12, 12))
sns.heatmap(test_df.isna())
test_df['new text'] = test_df['text'] + ' ' + test_df['location'] + ' ' + test_df['keyword']
test_df['new text'] = test_df['new text'].str.replace('#', ' ')
test_df.head()
train_df.head()
test_df.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
X_train = train_df['new text']
y_train = train_df['target']
X_test = test_df['new text']
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
nb = MultinomialNB()