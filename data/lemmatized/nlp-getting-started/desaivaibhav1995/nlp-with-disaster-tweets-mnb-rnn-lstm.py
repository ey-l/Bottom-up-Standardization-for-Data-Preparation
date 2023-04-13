import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_id = _input1['id']
test_id = _input0['id']
_input1 = _input1.drop(columns=['id'], inplace=False)
_input0 = _input0.drop(columns=['id'], inplace=False)
_input1.isnull().sum()
_input0.isnull().sum()
_input1 = _input1.drop(columns=['keyword', 'location'], inplace=False)
_input0 = _input0.drop(columns=['keyword', 'location'], inplace=False)
_input1.head()
_input1['text'] = [t.lower() for t in _input1['text']]
_input0['text'] = [t.lower() for t in _input0['text']]
import re
import string
_input1['text'] = [re.sub('[%s]' % re.escape(string.punctuation), '', i) for i in _input1['text']]
_input0['text'] = [re.sub('[%s]' % re.escape(string.punctuation), '', i) for i in _input0['text']]
_input1['text'] = [re.sub('\\d', '', n) for n in _input1['text']]
_input0['text'] = [re.sub('\\d', '', n) for n in _input0['text']]
import nltk
from nltk.tokenize import word_tokenize
_input1['text'] = [word_tokenize(i) for i in _input1['text']]
_input0['text'] = [word_tokenize(i) for i in _input0['text']]
_input1['text'].head()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
_input1['text'] = [[i for i in j if not i in stop_words] for j in _input1['text']]
_input0['text'] = [[i for i in j if not i in stop_words] for j in _input0['text']]
_input1.head()
from collections import defaultdict
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
tag_map
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
_input1['text'] = [[lemmatizer.lemmatize(word, tag_map[tag[0]]) for (word, tag) in pos_tag(i)] for i in _input1['text']]
_input0['text'] = [[lemmatizer.lemmatize(word, tag_map[tag[0]]) for (word, tag) in pos_tag(i)] for i in _input0['text']]
_input1.head()
_input1['lemmatized_text'] = _input1['text'].apply(lambda x: ' '.join(x))
_input0['lemmatized_text'] = _input0['text'].apply(lambda x: ' '.join(x))
_input1.head()
_input1 = _input1.drop(columns=['text'], inplace=False)
_input0 = _input0.drop(columns=['text'], inplace=False)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
train_emb = tfidf.fit_transform(_input1['lemmatized_text']).toarray()
test_emb = tfidf.fit_transform(_input0['lemmatized_text']).toarray()
train_emb.shape[1:]
y = _input1['target']
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
MNB = MultinomialNB()
(x_train, x_valid, y_train, y_valid) = train_test_split(train_emb, y, test_size=0.3, random_state=100)