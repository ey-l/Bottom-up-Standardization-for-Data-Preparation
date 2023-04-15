import numpy as np
import pandas as pd
from sklearn import feature_extraction, model_selection, preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_df.head(2)
test_df.tail(2)

def transform_link(text):
    return ['http' if i.startswith('http') else i for i in text]

def preprocess_text(data):
    rows = []
    for text in data:
        text = text.lower()
        text = text.translate(text.maketrans('', '', string.punctuation))
        tokens = re.split('\\W+', text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        tokens = transform_link(tokens)
        stemmer = PorterStemmer()
        row = [stemmer.stem(token) for token in tokens]
        rows.append(' '.join(row))
    return rows
text = train_df['text'][-1:]
tokens = preprocess_text(text)
print(text)
print(tokens)
(X_train, y_train) = (preprocess_text(train_df['text']), train_df['target'])
X_test = preprocess_text(test_df['text'])
count_vectorizer = feature_extraction.text.CountVectorizer()
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
param_grid = {'min_samples_split': [35, 40, 45], 'min_samples_leaf': [0.5, 1, 2], 'max_features': ['auto', 'sqrt']}
clf = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='f1')