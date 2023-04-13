import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(_input1.isnull())
sns.heatmap(_input0.isnull())
_input1 = _input1.drop(['id', 'location', 'keyword'], axis=1)
_input0 = _input0.drop(['id', 'location', 'keyword'], axis=1)
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string

def text_cleaning(text):
    """
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    """
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text
_input1['text'] = _input1['text'].apply(text_cleaning)
_input0['text'] = _input0['text'].apply(text_cleaning)
_input1
x = _input1['text']
y = _input1['target']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=0, test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=2, ngram_range=(1, 2))
x_train_trans = cv.fit_transform(x_train)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500)