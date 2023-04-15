import numpy as np
import pandas as pd
import plotly.express as px
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
xtrain = pd.read_csv('data/input/nlp-getting-started/train.csv')
xtest = pd.read_csv('data/input/nlp-getting-started/test.csv')
sample = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
xtrain.head()
y = xtrain['target']
ids = xtest['id']
xtrain.drop(['id', 'target'], inplace=True, axis=1)
xtest.drop('id', inplace=True, axis=1)
xtrain
xtrain.isnull().sum()
top_locations = xtrain['location'].value_counts().index.to_list()[:15]
top_loc_df = xtrain[xtrain.location.isin(top_locations)]
top_loc_df.head()
px.histogram(top_loc_df, x='location', color='location')
xtrain['location'].fillna(xtrain['location'].mode()[0], inplace=True)
xtest['location'].fillna(xtrain['location'].mode()[0], inplace=True)
top_keywords = xtrain['keyword'].value_counts().index.to_list()[:15]
top_keywords_df = xtrain[xtrain.keyword.isin(top_keywords)]
top_keywords_df.head()
px.histogram(top_keywords_df, x='keyword', color='keyword')
xtrain['keyword'].fillna(xtrain['keyword'].mode()[0], inplace=True)
xtest['keyword'].fillna(xtrain['keyword'].mode()[0], inplace=True)
xtrain.isnull().sum()
xtrain['text'] = xtrain.text.str.replace('[^a-zA-Z0-9\\s#]', '', regex=True)
xtrain['location'] = xtrain.location.str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
xtrain['keyword'] = xtrain.keyword.str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
xtest['text'] = xtest.text.str.replace('[^a-zA-Z0-9\\s#]', '', regex=True)
xtest['location'] = xtest.location.str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
xtest['keyword'] = xtest.keyword.str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
xtrain['location'] = xtrain['location'].str.strip()
xtest['location'] = xtest['location'].str.strip()
xtrain.text[0]
tfidf = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
train_texts = tfidf.fit_transform(xtrain['text'])
train_texts = pd.DataFrame(train_texts.toarray(), columns=tfidf.get_feature_names_out())
test_texts = tfidf.fit_transform(xtest['text'])
test_texts = pd.DataFrame(test_texts.toarray(), columns=tfidf.get_feature_names_out())
(train_texts, test_texts) = train_texts.align(test_texts, join='left', axis=1)
test_texts.fillna(0, inplace=True)
test_texts.isnull().sum().sum()
ohe_train_location = pd.get_dummies(xtrain['location'])
ohe_test_location = pd.get_dummies(xtest['location'])
(train_locations, test_locations) = ohe_train_location.align(ohe_test_location, join='left', axis=1)
test_locations.fillna(0, inplace=True)
test_locations.isnull().sum().sum()
ohe_train_keywords = pd.get_dummies(xtrain['keyword'])
ohe_test_keywords = pd.get_dummies(xtest['keyword'])
(train_keywords, test_keywords) = ohe_train_location.align(ohe_test_location, join='left', axis=1)
test_keywords.fillna(0, inplace=True)
test_keywords.isnull().sum().sum()
final_train_df = pd.concat([train_texts, train_locations, train_keywords], axis=1)
final_test_df = pd.concat([test_texts, test_locations, test_keywords], axis=1)
final_train_df
model = RandomForestClassifier(n_jobs=-1, random_state=1)