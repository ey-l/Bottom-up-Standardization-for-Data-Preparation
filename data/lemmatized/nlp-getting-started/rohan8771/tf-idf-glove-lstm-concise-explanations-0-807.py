import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def make_lower(text):
    return text.lower()

def remove_urls(text):
    url = 'https?://\\S+|www\\.\\S+'
    return re.sub(url, '', text)

def remove_html(text):
    html = '<.*?>'
    return re.sub(html, '', text)

def remove_mentions(text):
    mention = '@[A-Za-z0-9_]+'
    return re.sub(mention, '', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    return ' '.join(new_text)
porter = PorterStemmer()

def do_stemming(text):
    new_text = [porter.stem(word) for word in text.split()]
    return ' '.join(new_text)
_input1['text'] = _input1['text'].apply(lambda text: make_lower(text))
_input1['text'] = _input1['text'].apply(lambda text: remove_urls(text))
_input1['text'] = _input1['text'].apply(lambda text: remove_html(text))
_input1['text'] = _input1['text'].apply(lambda text: remove_mentions(text))
_input1['text'] = _input1['text'].apply(lambda text: remove_punct(text))
_input1['text'] = _input1['text'].apply(lambda text: remove_stopwords(text))
_input1['text'] = _input1['text'].apply(lambda text: do_stemming(text))
_input1.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectorized_text_df = pd.DataFrame(tfidf_vectorizer.fit_transform(_input1['text']).toarray())
train_tfidf = pd.concat([_input1, tfidf_vectorized_text_df], axis=1)
train_tfidf.head()
train_tfidf = train_tfidf.drop('text', axis=1, inplace=False)
train_tfidf['keyword'].value_counts().to_frame().reset_index().head().rename(columns={'index': 'Keyword', 'keyword': 'Frequency'})
_input1.groupby('keyword')['target'].mean().to_frame().reset_index().sort_values(by=['target'], ascending=[False]).head(10).rename(columns={'keyword': 'Keyword', 'target': 'Target Proportion'})
keywords_with_high_chance_of_disaster = _input1.groupby('keyword')['target'].mean().to_frame().reset_index().sort_values(by=['target'], ascending=[False])[:25]['keyword'].tolist()
train_tfidf['keyword'] = train_tfidf['keyword'].fillna('n', inplace=False)
train_tfidf['keyword_high_chance'] = train_tfidf['keyword'].apply(lambda keyword: 1 if keyword in keywords_with_high_chance_of_disaster else 0)
train_tfidf = train_tfidf.drop('keyword', axis=1, inplace=False)
train_tfidf.head()
train_tfidf['location'].value_counts().to_frame().reset_index().rename(columns={'index': 'Location', 'location': 'Frequency'}).head(10)
high_freq_locs = train_tfidf['location'].value_counts().to_frame().reset_index().rename(columns={'index': 'Location', 'location': 'Frequency'}).head(10)['Location'].tolist()
high_freq_locs
train_tfidf[train_tfidf['location'].isin(high_freq_locs)].groupby('location')['target'].mean().to_frame().reset_index().sort_values(by=['target'], ascending=[False])
train_tfidf['location_high_chance'] = train_tfidf['location'].apply(lambda location: 1 if location in ['Mumbai', 'India', 'Nigeria'] else 0)
train_tfidf.head()
train_tfidf = train_tfidf.drop('id', axis=1, inplace=False)
train_tfidf = train_tfidf.drop('location', axis=1, inplace=False)
train_tfidf['num_words'] = _input1['text'].apply(lambda text: len(text.split()))
import matplotlib.pyplot as plt
import seaborn as sns
sns.kdeplot(data=train_tfidf, x='num_words', hue='target')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(train_tfidf.drop('target', axis=1), train_tfidf['target'], random_state=42)
(X_train.shape, X_val.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
knn = KNeighborsClassifier(n_neighbors=5)