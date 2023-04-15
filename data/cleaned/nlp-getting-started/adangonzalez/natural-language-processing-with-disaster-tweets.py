import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
trainPath = 'data/input/nlp-getting-started/train.csv'
testPath = 'data/input/nlp-getting-started/test.csv'
train_df = pd.read_csv(trainPath)
test_df = pd.read_csv(testPath)
train_df.head()
train_df.describe().T
(fig, ax) = plt.subplots(figsize=(6, 10))
train_df['target'].value_counts(sort=False).plot(kind='bar')
x = ['Disaster', 'Not a disaster']
default_x_ticks = range(len(x))
plt.xticks(default_x_ticks, x, rotation=0)
plt.xlabel('Target')
plt.ylabel('Number of tweets')

import plotly.express as px
fig = px.pie(train_df, names='target', height=600, width=600, title='Pie Chart for distribution of Tweets')
fig.update_traces(textfont_size=15)
fig.show()
(fig, axes) = plt.subplots(figsize=(12, 10))
top_keywords = train_df['keyword'].value_counts()[:20]
ax = sns.barplot(y=top_keywords.index, x=top_keywords, data=train_df)
for container in ax.containers:
    ax.bar_label(container)
plt.title('Top 20 Keywords in tweets', fontsize=15)

train_df['length'] = train_df['text'].apply(len)
plt.figure(figsize=(10, 10))
sns.histplot(train_df['length'], kde=True)
plt.title('Length of the tweets')
plt.xlabel('Number of characters')
plt.ylabel('Thickness')
(fig, axes) = plt.subplots(figsize=(12, 10))
top_location = train_df['location'].value_counts()[:20]
ax = sns.barplot(y=top_location.index, x=top_location, palette='tab10', data=train_df)
for container in ax.containers:
    ax.bar_label(container)
plt.title('Top 20 locations', fontsize=15)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = feature_extraction.text.TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_df['text'])
test_vectors = vectorizer.transform(test_df['text'])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df['target'], cv=3, scoring='f1')
scores