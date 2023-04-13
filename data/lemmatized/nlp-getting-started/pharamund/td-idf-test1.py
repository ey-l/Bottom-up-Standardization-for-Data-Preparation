import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
print(_input1.shape)
print(_input1['target'].unique())
_input1.head()
train_text = _input1['text']
test_text = _input0['text']