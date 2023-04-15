from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
X = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col='id')
y = X.target
X.drop(['target'], axis=1, inplace=True)
X_test = pd.read_csv('data/input/nlp-getting-started/test.csv', index_col='id')
target_col = 'target'
text_cols = ['text']
categorical_cols = ['keyword', 'location']
X
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import re
stopwords = stopwords.words('english') + ["'d", "'ll", "'re", "'s", "'ve", 'could', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would']
cv = CountVectorizer(ngram_range=(1, 1), tokenizer=word_tokenize)
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y.values, train_size=0.8, test_size=0.2, random_state=0)
smart_vectorized_x_train = cv.fit_transform(X_train.text).toarray()
catboost_params = {'iterations': 1000, 'learning_rate': 0.1, 'eval_metric': 'Logloss', 'early_stopping_rounds': 100, 'use_best_model': True, 'verbose': 300}
clf = CatBoostClassifier(**catboost_params)
smart_vectorized_x_valid = cv.transform(X_valid.text).toarray()