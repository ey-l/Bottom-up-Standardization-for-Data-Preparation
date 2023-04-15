import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
from sklearn.model_selection import train_test_split
train_sentences = train_df['text'].to_numpy()
train_labels = train_df['target'].to_numpy()
model_0 = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])