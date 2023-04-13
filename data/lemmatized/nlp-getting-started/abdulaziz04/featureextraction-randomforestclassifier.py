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
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1.head()
y = _input1['target']
ids = _input0['id']
_input1 = _input1.drop(['id', 'target'], inplace=False, axis=1)
_input0 = _input0.drop('id', inplace=False, axis=1)
_input1
_input1.isnull().sum()
top_locations = _input1['location'].value_counts().index.to_list()[:15]
top_loc_df = _input1[_input1.location.isin(top_locations)]
top_loc_df.head()
px.histogram(top_loc_df, x='location', color='location')
_input1['location'] = _input1['location'].fillna(_input1['location'].mode()[0], inplace=False)
_input0['location'] = _input0['location'].fillna(_input1['location'].mode()[0], inplace=False)
top_keywords = _input1['keyword'].value_counts().index.to_list()[:15]
top_keywords_df = _input1[_input1.keyword.isin(top_keywords)]
top_keywords_df.head()
px.histogram(top_keywords_df, x='keyword', color='keyword')
_input1['keyword'] = _input1['keyword'].fillna(_input1['keyword'].mode()[0], inplace=False)
_input0['keyword'] = _input0['keyword'].fillna(_input1['keyword'].mode()[0], inplace=False)
_input1.isnull().sum()
_input1['text'] = _input1.text.str.replace('[^a-zA-Z0-9\\s#]', '', regex=True)
_input1['location'] = _input1.location.str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
_input1['keyword'] = _input1.keyword.str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
_input0['text'] = _input0.text.str.replace('[^a-zA-Z0-9\\s#]', '', regex=True)
_input0['location'] = _input0.location.str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
_input0['keyword'] = _input0.keyword.str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
_input1['location'] = _input1['location'].str.strip()
_input0['location'] = _input0['location'].str.strip()
_input1.text[0]
tfidf = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
train_texts = tfidf.fit_transform(_input1['text'])
train_texts = pd.DataFrame(train_texts.toarray(), columns=tfidf.get_feature_names_out())
test_texts = tfidf.fit_transform(_input0['text'])
test_texts = pd.DataFrame(test_texts.toarray(), columns=tfidf.get_feature_names_out())
(train_texts, test_texts) = train_texts.align(test_texts, join='left', axis=1)
test_texts = test_texts.fillna(0, inplace=False)
test_texts.isnull().sum().sum()
ohe_train_location = pd.get_dummies(_input1['location'])
ohe_test_location = pd.get_dummies(_input0['location'])
(train_locations, test_locations) = ohe_train_location.align(ohe_test_location, join='left', axis=1)
test_locations = test_locations.fillna(0, inplace=False)
test_locations.isnull().sum().sum()
ohe_train_keywords = pd.get_dummies(_input1['keyword'])
ohe_test_keywords = pd.get_dummies(_input0['keyword'])
(train_keywords, test_keywords) = ohe_train_location.align(ohe_test_location, join='left', axis=1)
test_keywords = test_keywords.fillna(0, inplace=False)
test_keywords.isnull().sum().sum()
final_train_df = pd.concat([train_texts, train_locations, train_keywords], axis=1)
final_test_df = pd.concat([test_texts, test_locations, test_keywords], axis=1)
final_train_df
model = RandomForestClassifier(n_jobs=-1, random_state=1)