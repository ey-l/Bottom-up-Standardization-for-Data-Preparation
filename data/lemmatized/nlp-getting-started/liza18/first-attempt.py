import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col=0)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv', index_col=0)
_input1.head(5)
_input1.describe()
_input1.isnull().sum()
_input1 = _input1.drop(['keyword', 'location'], axis=1, inplace=False)
_input0 = _input0.drop(['keyword', 'location'], axis=1, inplace=False)
_input1.head(5)
_input1.duplicated().sum()
_input1 = _input1.drop_duplicates(inplace=False)
_input1.info()
_input1.shape
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
plt.bar(['Disaster', 'NotDisaster'], [(_input1.target == 1).sum(), (_input1.target == 0).sum()])
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
_input1['text'] = _input1['text'].str.replace('^.+@[^\\.].*\\.[a-z]{2,}$', 'emailaddress')
_input1['text'] = _input1['text'].str.replace('^http\\://[a-zA-Z0-9\\-\\.]+\\.[a-zA-Z]{2,3}(/\\S*)?$', 'webaddress')
_input1['text'] = _input1['text'].str.replace('£|\\$', 'money-symbol')
_input1['text'] = _input1['text'].str.replace('^\\(?[\\d]{3}\\)?[\\s-]?[\\d]{3}[\\s-]?[\\d]{4}$', 'phone-number')
_input1['text'] = _input1['text'].str.replace('\\d+(\\.\\d+)?', 'number')
_input1['text'] = _input1['text'].str.replace('[^\\w\\d\\s]', ' ')
_input1['text'] = _input1['text'].str.replace('\\s+', ' ')
_input1['text'] = _input1['text'].str.replace('^\\s+|\\s*?$', ' ')
_input1['text'] = _input1['text'].str.lower()
nltk.download('popular')
stop_words = set(stopwords.words('english'))
_input1['text'] = _input1['text'].apply(lambda x: ' '.join((term for term in x.split() if term not in stop_words)))
from nltk.stem import PorterStemmer, LancasterStemmer
ss = nltk.SnowballStemmer('english')
_input1['text'] = _input1['text'].apply(lambda x: ' '.join((ss.stem(term) for term in x.split())))

def dictionary(check):
    check = check.str.extractall('([a-zA_Z]+)')
    check.columns = ['check']
    b = check.reset_index(drop=True)
    check = b['check'].value_counts()
    dictionary = pd.DataFrame({'word': check.index, 'freq': check.values})
    dictionary.index = dictionary['word']
    dictionary = dictionary.drop('word', axis=1, inplace=False)
    dictionary = dictionary.sort_values('freq', inplace=False, ascending=False)
    return dictionary
dictionary_clean = dictionary(_input1['text'])
dictionary_clean[:20].plot(kind='barh', figsize=(10, 10))
pd.DataFrame(_input1['target'].value_counts() / _input1.shape[0] * 100).round(2)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
(X_train, X_test, y_train, y_test) = train_test_split(_input1.text, _input1.target, test_size=0.3, stratify=_input1.target, random_state=1672)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', stop_words='english', token_pattern='\\w{1,}')
train_tfidf = tfidf.fit_transform(X_train)
test_tfidf = tfidf.transform(X_test)
_input0 = tfidf.transform(_input0.text)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
clf = MultinomialNB(alpha=1)
scores = cross_val_score(clf, train_tfidf, y_train, cv=5, scoring='f1')
scores