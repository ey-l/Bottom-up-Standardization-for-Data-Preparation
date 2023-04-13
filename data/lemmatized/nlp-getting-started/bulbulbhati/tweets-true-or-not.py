import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1.isnull().sum()
_input1.info
_input0.isnull().sum()
_input0.info
_input1 = _input1.drop(columns='location', inplace=False)
_input0 = _input0.drop(columns='location', inplace=False)
_input1[_input1['keyword'].notnull()][_input1['target'] == 1]
_input1[_input1['keyword'].notnull()][_input1['target'] == 0]
_input1['keyword'].value_counts().index
_input1.head(10)

def lowercase_text(text):
    text = text.lower()
    return text
_input1['text'] = _input1['text'].apply(lambda x: lowercase_text(x))
_input0['text'] = _input0['text'].apply(lambda x: lowercase_text(x))
_input1['text'].head(10)
import string
string.punctuation
_input1.head(10)

def remove_punctuation(text):
    text_no_punctuation = ''.join([c for c in text if c not in string.punctuation])
    return text_no_punctuation
_input1['text'] = _input1['text'].apply(lambda x: remove_punctuation(x))
_input0['text'] = _input0['text'].apply(lambda x: remove_punctuation(x))
_input1['text'].head(10)
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = nltk.tokenize.RegexpTokenizer('\\w+')
_input1['text'] = _input1['text'].apply(lambda x: tokenizer.tokenize(x))
_input0['text'] = _input0['text'].apply(lambda x: tokenizer.tokenize(x))
_input1['text'].head()
from nltk.corpus import stopwords
print(stopwords.words('english'))
_input1.head(10)

def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    words = [w for w in text if w not in stopwords.words('english')]
    return words
_input1['text'] = _input1['text'].apply(lambda x: remove_stopwords(x))
_input0['text'] = _input0['text'].apply(lambda x: remove_stopwords(x))
_input1.head(10)

def combine_text(list_of_text):
    combine_text = ' '.join(list_of_text)
    return combine_text
_input1['text'] = _input1['text'].apply(lambda x: combine_text(x))
_input0['text'] = _input0['text'].apply(lambda x: combine_text(x))
_input1.head()
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
train_vector = count_vectorizer.fit_transform(_input1['text']).todense()
test_vector = count_vectorizer.transform(_input0['text']).todense()
print(count_vectorizer.vocabulary_)
print(train_vector.shape)
print(test_vector.shape)
from sklearn.model_selection import train_test_split
Y = _input1['target']
(x_train, x_test, y_train, y_test) = train_test_split(train_vector, Y, test_size=0.3, random_state=0)
y_train
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=3.0)
scores = cross_val_score(model, train_vector, _input1['target'], cv=5)
print(scores.mean())