import numpy as np
import pandas as pd
import os
train_set = pd.read_csv('data/input/nlp-getting-started/train.csv')
print(len(train_set))
test_set = pd.read_csv('data/input/nlp-getting-started/test.csv')
X = train_set['text'].values
location = train_set[['location', 'keyword']].fillna('').values
y = train_set['target'].values
X_test = test_set['text'].values
location_test = test_set[['location', 'keyword']].fillna('').values
ids = test_set['id']
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')
ps = PorterStemmer()
lemma = WordNetLemmatizer()
disasters = ['storm', 'eartquake', 'fire', 'hurricane', 'tornado', 'flood', 'flame', 'volcano', 'suicide']

def clean_word(word):
    word = re.sub('[,#$%^|&*@/\\ûûª]', '', word)
    word = re.sub('[-_;!?:=().]', ' ', word)
    word = word.replace("'", ' ')
    word = word.replace('\x89û', '')
    word = re.sub('\\.+', '.', word)
    word = re.sub('[0-9/]', '', word)
    word = word.lower()
    if len(word.split(' ')) > 1:
        words = [clean_word(w) for w in word.split(' ')]
        return ' '.join(words)
    word = lemma.lemmatize(word)
    if type(word) == list:
        word = word[0]
    return word

def clean_text(text):
    cleaned_text = []
    is_link = 0
    hastags = []
    for line in text.split('\n'):
        for word in line.split(' '):
            if 'http' in word:
                is_link += 1
                continue
            if word.startswith('#'):
                hastags.append(word.replace('#', ''))
            word = clean_word(word)
            cleaned_text.append(word)
    text = ' '.join(cleaned_text)
    hastags = ' '.join(hastags)
    return (text, is_link, hastags)
X = np.array([clean_text(xi) for xi in X])
mapping = np.vectorize(clean_word)
location = mapping(location)
location_test = mapping(location_test)
print(X.shape)
X_test = np.array([clean_text(xi) for xi in X_test])
from sklearn.model_selection import train_test_split
X_whole = np.append(X, location.reshape(len(location), 2), axis=1)
(train_x, val_x, train_y, val_y) = train_test_split(X_whole, y, test_size=0.05, stratify=y, random_state=0, shuffle=True)
print(train_x[:10], '\n', train_y[:10])
X_test = np.append(X_test, location_test.reshape(len(location_test), 2), axis=1)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=3, strip_accents='unicode')
vec_train_x = vectorizer.fit_transform(train_x[:, :-4].reshape(len(train_x))).toarray()
vec_val_x = vectorizer.transform(val_x[:, :-4].reshape(len(val_x))).toarray()
locs_train = train_x[:, -3:]
locs_val = val_x[:, -3:]
vectorizer_hastags = TfidfVectorizer(max_features=2000)
hashs_vecs_train = vectorizer_hastags.fit_transform(locs_train[:, -3]).toarray()
hashs_vecs_val = vectorizer_hastags.transform(locs_val[:, -3]).toarray()
vectorizer_locations = TfidfVectorizer(max_features=2000)
locs_vecs_train = vectorizer_locations.fit_transform(locs_train[:, -2]).toarray()
locs_vecs_val = vectorizer_locations.transform(locs_val[:, -2]).toarray()
vectorizer_keywords = TfidfVectorizer(max_features=2000)
keywords_train = vectorizer_keywords.fit_transform(locs_train[:, -1]).toarray()
keywords_val = vectorizer_keywords.transform(locs_val[:, -1]).toarray()
print(vec_train_x.shape, locs_vecs_train.shape)
vec_train_x = np.append(vec_train_x, train_x[:, -4].astype(np.float32).reshape(len(train_x), 1), axis=1)
vec_val_x = np.append(vec_val_x, val_x[:, -4].astype(np.float32).reshape(len(val_x), 1), axis=1)
print(vec_train_x.shape)
vec_train_x = csr_matrix(np.append(np.append(np.append(vec_train_x, locs_vecs_train, axis=1), keywords_train, axis=1), hashs_vecs_train, axis=1))
vec_val_x = csr_matrix(np.append(np.append(np.append(vec_val_x, locs_vecs_val, axis=1), keywords_val, axis=1), hashs_vecs_val, axis=1))
model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.8, min_samples_split=50, random_state=0)