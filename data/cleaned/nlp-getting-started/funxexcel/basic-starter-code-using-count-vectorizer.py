import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_df['text'])
test_vectors = count_vectorizer.transform(test_df['text'])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df['target'], cv=3, scoring='f1')
scores.mean()