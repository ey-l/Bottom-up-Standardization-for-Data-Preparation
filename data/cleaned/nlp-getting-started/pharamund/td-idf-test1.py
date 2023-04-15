import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
print(train.shape)
print(train['target'].unique())
train.head()
train_text = train['text']
test_text = test['text']