import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
import re

def clean(text):
    res = re.sub('http(s)?:\\/\\/([\\w\\.\\/])*', ' ', text)
    res = re.sub('[0-9]+', '', res)
    res = re.sub('[!"#$%&()*+,-./:;=?@\\\\^_`"~\\t\\n\\<\\>\\[\\]\\{\\}]', ' ', res)
    res = re.sub('  +', ' ', res)
    return res.strip()
_input1['text'] = _input1['text'].apply(clean)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(max_iter=2000, tol=0.0005))])
param_grid = {'clf__max_iter': [2000, 3000, 4000], 'clf__tol': [0.01, 0.001, 0.0001]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)