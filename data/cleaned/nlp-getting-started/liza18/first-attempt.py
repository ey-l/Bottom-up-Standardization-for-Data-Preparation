import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col=0)
test = pd.read_csv('data/input/nlp-getting-started/test.csv', index_col=0)
train.head(5)
train.describe()
train.isnull().sum()
train.drop(['keyword', 'location'], axis=1, inplace=True)
test.drop(['keyword', 'location'], axis=1, inplace=True)
train.head(5)
train.duplicated().sum()
train.drop_duplicates(inplace=True)
train.info()
train.shape
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
plt.bar(['Disaster', 'NotDisaster'], [(train.target == 1).sum(), (train.target == 0).sum()])

import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
train['text'] = train['text'].str.replace('^.+@[^\\.].*\\.[a-z]{2,}$', 'emailaddress')
train['text'] = train['text'].str.replace('^http\\://[a-zA-Z0-9\\-\\.]+\\.[a-zA-Z]{2,3}(/\\S*)?$', 'webaddress')
train['text'] = train['text'].str.replace('£|\\$', 'money-symbol')
train['text'] = train['text'].str.replace('^\\(?[\\d]{3}\\)?[\\s-]?[\\d]{3}[\\s-]?[\\d]{4}$', 'phone-number')
train['text'] = train['text'].str.replace('\\d+(\\.\\d+)?', 'number')
train['text'] = train['text'].str.replace('[^\\w\\d\\s]', ' ')
train['text'] = train['text'].str.replace('\\s+', ' ')
train['text'] = train['text'].str.replace('^\\s+|\\s*?$', ' ')
train['text'] = train['text'].str.lower()
nltk.download('popular')
stop_words = set(stopwords.words('english'))
train['text'] = train['text'].apply(lambda x: ' '.join((term for term in x.split() if term not in stop_words)))
from nltk.stem import PorterStemmer, LancasterStemmer
ss = nltk.SnowballStemmer('english')
train['text'] = train['text'].apply(lambda x: ' '.join((ss.stem(term) for term in x.split())))

def dictionary(check):
    check = check.str.extractall('([a-zA_Z]+)')
    check.columns = ['check']
    b = check.reset_index(drop=True)
    check = b['check'].value_counts()
    dictionary = pd.DataFrame({'word': check.index, 'freq': check.values})
    dictionary.index = dictionary['word']
    dictionary.drop('word', axis=1, inplace=True)
    dictionary.sort_values('freq', inplace=True, ascending=False)
    return dictionary
dictionary_clean = dictionary(train['text'])
dictionary_clean[:20].plot(kind='barh', figsize=(10, 10))
pd.DataFrame(train['target'].value_counts() / train.shape[0] * 100).round(2)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
(X_train, X_test, y_train, y_test) = train_test_split(train.text, train.target, test_size=0.3, stratify=train.target, random_state=1672)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', stop_words='english', token_pattern='\\w{1,}')
train_tfidf = tfidf.fit_transform(X_train)
test_tfidf = tfidf.transform(X_test)
test = tfidf.transform(test.text)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
clf = MultinomialNB(alpha=1)
scores = cross_val_score(clf, train_tfidf, y_train, cv=5, scoring='f1')
scores