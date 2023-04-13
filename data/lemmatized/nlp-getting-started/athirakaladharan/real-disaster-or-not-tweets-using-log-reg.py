import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.shape
_input1.isnull().sum()
_input1 = _input1.drop('keyword', axis=1, inplace=False)
_input1 = _input1.drop('location', axis=1, inplace=False)
_input1.target.nunique()
_input1.target.value_counts()
_input1['text'][100]
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
_input1.text = _input1.text.str.lower()
from nltk import *
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), tokenizer=LemmaTokenizer())
x = tfidf.fit_transform(_input1.text.to_list())
X = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
X.head()
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(_input1.target)
y
enc.classes_
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=33)
X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()