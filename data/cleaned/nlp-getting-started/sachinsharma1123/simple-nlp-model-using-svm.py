import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
submission = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_df
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(train_df.isnull())
sns.heatmap(test_df.isnull())
train_df = train_df.drop(['id', 'location', 'keyword'], axis=1)
test_df = test_df.drop(['id', 'location', 'keyword'], axis=1)
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
train_df['text'] = train_df['text'].apply(text_cleaning)
test_df['text'] = test_df['text'].apply(text_cleaning)
train_df
x = train_df['text']
y = train_df['target']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=0, test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=2, ngram_range=(1, 2))
x_train_trans = cv.fit_transform(x_train)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500)