import numpy as pn
import pandas as pd
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import model_selection
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
cv = feature_extraction.text.CountVectorizer()
train_v = cv.fit_transform(_input1['text'])
test_v = cv.transform(_input0['text'])
model = linear_model.RidgeClassifier(alpha=20.0)
scores = model_selection.cross_val_score(model, train_v, _input1['target'], cv=3, scoring='f1')
print(f'Accuracy: {scores}')