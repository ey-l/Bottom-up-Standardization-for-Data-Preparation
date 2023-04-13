import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
from sklearn.model_selection import train_test_split
train_sentences = _input1['text'].to_numpy()
train_labels = _input1['target'].to_numpy()
model_0 = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])