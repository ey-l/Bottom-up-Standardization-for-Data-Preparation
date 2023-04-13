import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1 = _input1.drop('id', axis=1)
_input1 = _input1.drop('keyword', axis=1)
_input1 = _input1.drop('location', axis=1)
_input0 = _input0.drop('id', axis=1)
_input0 = _input0.drop('keyword', axis=1)
_input0 = _input0.drop('location', axis=1)
count_vectorizer = feature_extraction.text.TfidfVectorizer()
train_vectors = count_vectorizer.fit_transform(_input1['text'])
test_vectors = count_vectorizer.transform(_input0['text'])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, _input1['target'], cv=3, scoring='f1')
scores