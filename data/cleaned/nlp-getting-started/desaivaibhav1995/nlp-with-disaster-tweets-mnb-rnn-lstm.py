import pandas as pd
import numpy as np
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_id = train['id']
test_id = test['id']
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)
train.isnull().sum()
test.isnull().sum()
train.drop(columns=['keyword', 'location'], inplace=True)
test.drop(columns=['keyword', 'location'], inplace=True)
train.head()
train['text'] = [t.lower() for t in train['text']]
test['text'] = [t.lower() for t in test['text']]
import re
import string
train['text'] = [re.sub('[%s]' % re.escape(string.punctuation), '', i) for i in train['text']]
test['text'] = [re.sub('[%s]' % re.escape(string.punctuation), '', i) for i in test['text']]
train['text'] = [re.sub('\\d', '', n) for n in train['text']]
test['text'] = [re.sub('\\d', '', n) for n in test['text']]
import nltk
from nltk.tokenize import word_tokenize
train['text'] = [word_tokenize(i) for i in train['text']]
test['text'] = [word_tokenize(i) for i in test['text']]
train['text'].head()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
train['text'] = [[i for i in j if not i in stop_words] for j in train['text']]
test['text'] = [[i for i in j if not i in stop_words] for j in test['text']]
train.head()
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
train['text'] = [[lemmatizer.lemmatize(word, tag_map[tag[0]]) for (word, tag) in pos_tag(i)] for i in train['text']]
test['text'] = [[lemmatizer.lemmatize(word, tag_map[tag[0]]) for (word, tag) in pos_tag(i)] for i in test['text']]
train.head()
train['lemmatized_text'] = train['text'].apply(lambda x: ' '.join(x))
test['lemmatized_text'] = test['text'].apply(lambda x: ' '.join(x))
train.head()
train.drop(columns=['text'], inplace=True)
test.drop(columns=['text'], inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
train_emb = tfidf.fit_transform(train['lemmatized_text']).toarray()
test_emb = tfidf.fit_transform(test['lemmatized_text']).toarray()
train_emb.shape[1:]
y = train['target']
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
MNB = MultinomialNB()
(x_train, x_valid, y_train, y_valid) = train_test_split(train_emb, y, test_size=0.3, random_state=100)