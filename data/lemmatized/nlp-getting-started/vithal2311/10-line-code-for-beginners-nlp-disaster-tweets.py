from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
X_train = _input1.iloc[:, 3]
X_train.head()
y_train = _input1.target
y_train.head()
X_test = _input0.iloc[:, 3]
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.values)
count_train
count_test = count_vectorizer.transform(X_test.values)
count_test
nb_classifier = MultinomialNB()