from numpy.lib.function_base import median
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col=0)
_input1.info()
import re
import string
url_re = re.compile('https?://\\S+|www\\.\\S+')
tag_re = re.compile('<.*?>')
table_punct = str.maketrans('', '', string.punctuation)

def text_preprocess(text):
    text = url_re.sub('', text)
    text = tag_re.sub('', text)
    text = text.translate(table_punct)
    return text
texts = _input1['text'].apply(text_preprocess)
target = _input1['target']
(X_train, X_test, y_train, y_test) = train_test_split(texts, target, train_size=0.25)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
pipe = Pipeline([('vect', CountVectorizer()), ('tsvd', TruncatedSVD()), ('clf', RidgeClassifier())])