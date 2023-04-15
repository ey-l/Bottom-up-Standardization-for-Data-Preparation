import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import nltk
from nltk.corpus import stopwords
my_stopwords = stopwords.words('english')
import re
import spacy

nlp = spacy.load('en_core_web_sm')
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.head()
df.info()
df.describe(include='object')
sns.set_theme()
sns.countplot(df['target'])
df.isnull().sum()

def pp_remove_na(data):
    data['keyword'].fillna('unknown', inplace=True)
    data['location'].fillna('unknown', inplace=True)
    return data
df = pp_remove_na(df)
df[df.location != 'unknown']
print('Location cardinality: ', len(df['location'].unique()))
print('Keywords cardinality: ', len(df['keyword'].unique()))

def filter_cat(t_hold, serie):
    tot = serie.value_counts().sum()
    print('Initial category N:', len(serie.unique()))
    cat_list = []
    i = 0
    for cat in serie.value_counts().to_dict().items():
        if i < tot * t_hold:
            cat_list.append(cat[0])
            i += cat[1]
        else:
            print('Final category N:', len(cat_list))
            return cat_list
    print('None filtered')
    return cat_list
location_cat = filter_cat(0.5, df.location)
keyword_cat = filter_cat(0.5, df.keyword)

def pp_transform_categories(data, lc, kc):
    data.loc[~data['location'].isin(lc), 'location'] = 'other'
    data.loc[~data['keyword'].isin(kc), 'keyword'] = 'other'
    return data
df = pp_transform_categories(df, location_cat, keyword_cat)
print(len(df['keyword'].unique()))
print(len(df['location'].unique()))

def pp_lemmatize_text(data, nlp, sw):
    docs = []
    for i in range(0, len(data)):
        tweet = data['text'][i]
        tweet = tweet.lower()
        tweet = re.sub('https?://\\S+', '', tweet)
        tweet = re.sub('[^a-z0-9]', ' ', tweet)
        tweet = re.sub('\\n', ' ', tweet)
        tweet = re.sub('[ ]+', ' ', tweet)
        sentence = nlp(tweet)
        bs = [w.lemma_ for w in sentence if w.lemma_ not in sw]
        docs.append(' '.join(bs))
    print(docs[5])
    print(data['text'].iloc[5])
    Xdf = pd.concat([data[['location', 'keyword']], pd.DataFrame(docs, columns=['text'])], axis=1)
    return Xdf
Xdf = pp_lemmatize_text(df, nlp, my_stopwords)
Xdf
CV = CountVectorizer()
TFIDF = TfidfVectorizer()
OHE = OneHotEncoder()
X_text = CV.fit_transform(Xdf['text'])
X_text_tfidf = TFIDF.fit_transform(Xdf['text'])
X_cat = OHE.fit_transform(Xdf[['location', 'keyword']])
print(X_text.shape)
print(X_text_tfidf.shape)
print(X_cat.shape)
from scipy.sparse import hstack
X = hstack((X_cat, X_text))
X_tfidf = hstack((X_cat, X_text_tfidf))
X.shape
X_tfidf.shape
y = df['target'].to_numpy()
y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
(X_train_tfidf, X_test_tfidf, y_train, y_test) = train_test_split(X_tfidf, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier()