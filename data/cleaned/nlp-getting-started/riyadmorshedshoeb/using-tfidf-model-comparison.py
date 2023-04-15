import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
(X, y) = (train['text'], train['target'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()
len(vectorizer.get_feature_names())

def evaluate(y_true, y_predicted):
    acc = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    return (acc, precision, recall, f1)
from sklearn.linear_model import RidgeClassifier