from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pandas as pd
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
dt = pd.read_csv('data/input/nlp-getting-started/test.csv')
X_train = df.iloc[:, 3]
X_train.head()
y_train = df.target
y_train.head()
X_test = dt.iloc[:, 3]
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.values)
count_train
count_test = count_vectorizer.transform(X_test.values)
count_test
nb_classifier = MultinomialNB()