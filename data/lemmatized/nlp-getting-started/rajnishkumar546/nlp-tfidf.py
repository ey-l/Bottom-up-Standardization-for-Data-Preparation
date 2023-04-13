import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.shape
_input1.info()
_input1.isnull().sum()
_input1 = _input1.drop(columns=['keyword', 'location', 'id'], inplace=False)
_input1.head()
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)
_input1['transform_text'] = _input1['text'].apply(transform)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0 = _input0.drop(columns=['keyword', 'id', 'location'], inplace=False)
_input0.isnull().sum()
_input0.shape
_input0['transform_text'] = _input0['text'].apply(transform)
_input0 = _input0.drop(columns=['text'], inplace=False)
_input0.head()
_input0.shape
_input1.head()
_input1 = _input1.drop(columns=['text'], inplace=False)
_input1.shape
_input1 = pd.concat([_input1, _input0], axis=0)
_input1.head()
_input1.shape
_input1.isnull().sum()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tf = TfidfVectorizer()
X = tf.fit_transform(_input1['transform_text'])
X.shape
X_tests = X[7613:10876].toarray()
X_tests.shape
XX = X[0:7613].toarray()
XX.shape
y = _input1['target'].iloc[0:7613]
y[7612]
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(XX, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier(n_estimators=110)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
from sklearn.metrics import classification_report