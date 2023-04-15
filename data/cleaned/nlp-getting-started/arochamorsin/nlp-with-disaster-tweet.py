import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
import re
import string

def contraction(text):
    text = re.sub("won\\'t", 'will not', text)
    text = re.sub("can\\'t", 'can not', text)
    text = re.sub("n\\'t", ' not', text)
    text = re.sub("\\'re", ' are', text)
    text = re.sub("\\'s", ' is', text)
    text = re.sub("\\'d", ' would', text)
    text = re.sub("\\'ll", ' will', text)
    text = re.sub("\\'t", ' not', text)
    text = re.sub("\\'ve", ' have', text)
    text = re.sub("\\'m", ' am', text)
    return text
train['text'] = train['text'].apply(lambda x: contraction(x))
test['text'] = test['text'].apply(lambda x: contraction(x))
train['text'].head()

def urls(text):
    text = re.sub('^https?:\\/\\/.*[\\r\\n]*', '', text, flags=re.MULTILINE)
    return text
train['text'] = train['text'].apply(lambda x: urls(x))
test['text'] = test['text'].apply(lambda x: urls(x))
train['text'].head()

def spacial(text):
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text
train['text'] = train['text'].apply(lambda x: spacial(x))
test['text'] = test['text'].apply(lambda x: spacial(x))
train['text'].head()

def emoji(text):
    regrex_pattern = re.compile(pattern='[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿]+', flags=re.UNICODE)
    return regrex_pattern.sub('', text)
train['text'] = train['text'].apply(lambda x: emoji(x))
test['text'] = test['text'].apply(lambda x: emoji(x))
train['text'].head()
train.head()
test.head()
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
stopwords = stopwords.words('english')
count_vector = CountVectorizer(token_pattern='\\w{1,}', ngram_range=(1, 2), stop_words=stopwords)
from sklearn.model_selection import train_test_split
X = train.text
y = train.target
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
clf = LogisticRegression()
pipe = Pipeline([('count_vector', CountVectorizer()), ('clf', LogisticRegression())])