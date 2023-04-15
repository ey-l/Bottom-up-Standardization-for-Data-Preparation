import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
train.isnull().sum()
train.info
test.isnull().sum()
test.info
train.drop(columns='location', inplace=True)
test.drop(columns='location', inplace=True)
train[train['keyword'].notnull()][train['target'] == 1]
train[train['keyword'].notnull()][train['target'] == 0]
train['keyword'].value_counts().index
train.head(10)

def lowercase_text(text):
    text = text.lower()
    return text
train['text'] = train['text'].apply(lambda x: lowercase_text(x))
test['text'] = test['text'].apply(lambda x: lowercase_text(x))
train['text'].head(10)
import string
string.punctuation
train.head(10)

def remove_punctuation(text):
    text_no_punctuation = ''.join([c for c in text if c not in string.punctuation])
    return text_no_punctuation
train['text'] = train['text'].apply(lambda x: remove_punctuation(x))
test['text'] = test['text'].apply(lambda x: remove_punctuation(x))
train['text'].head(10)
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = nltk.tokenize.RegexpTokenizer('\\w+')
train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))
test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))
train['text'].head()
from nltk.corpus import stopwords
print(stopwords.words('english'))
train.head(10)

def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    words = [w for w in text if w not in stopwords.words('english')]
    return words
train['text'] = train['text'].apply(lambda x: remove_stopwords(x))
test['text'] = test['text'].apply(lambda x: remove_stopwords(x))
train.head(10)

def combine_text(list_of_text):
    combine_text = ' '.join(list_of_text)
    return combine_text
train['text'] = train['text'].apply(lambda x: combine_text(x))
test['text'] = test['text'].apply(lambda x: combine_text(x))
train.head()
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
train_vector = count_vectorizer.fit_transform(train['text']).todense()
test_vector = count_vectorizer.transform(test['text']).todense()
print(count_vectorizer.vocabulary_)
print(train_vector.shape)
print(test_vector.shape)
from sklearn.model_selection import train_test_split
Y = train['target']
(x_train, x_test, y_train, y_test) = train_test_split(train_vector, Y, test_size=0.3, random_state=0)
y_train
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=3.0)
scores = cross_val_score(model, train_vector, train['target'], cv=5)
print(scores.mean())