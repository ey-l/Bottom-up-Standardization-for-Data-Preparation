from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import string
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')

def cleaner(x):
    return [a for a in ''.join([a for a in x if a not in string.punctuation]).split() if a.lower() not in stopwords.words('english')]
lr = LogisticRegression(solver='liblinear', random_state=777)
Pipe = Pipeline([('bow', CountVectorizer(analyzer=cleaner)), ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])
_input1 = _input1.drop('id', inplace=False, axis=1)
test1 = _input0.drop('id', axis=1)
_input1 = _input1.fillna('None', inplace=False)
test1 = test1.fillna('None', inplace=False)
_input0 = _input0.fillna('None', inplace=False)
_input1