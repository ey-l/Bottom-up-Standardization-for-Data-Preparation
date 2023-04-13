import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
(_input1.shape, _input0.shape)
_input1.isna().sum()
_input1 = _input1.drop(['location'], axis=1)
_input1 = _input1.fillna(_input1.median(), inplace=False)
_input1 = _input1.drop(['keyword'], axis=1)
_input1.target.value_counts().plot(kind='bar', color='red')
import re
import nltk
from nltk.corpus import stopwords
Stop = stopwords.words('english')
stemmer = nltk.SnowballStemmer('english')

def newtweet(text):
    text = str(text).lower()
    text = re.sub('@\\s+|http\\s+|www.\\s+|\\n', '', text)
    text = re.sub('[^A-Za-z0-9\\s]+', '', text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = ' '.join([word for word in text if word not in Stop])
    text = text.strip()
    return text
_input1['text'] = _input1['text'].apply(newtweet)
_input0['text'] = _input0['text'].apply(newtweet)
_input0 = _input0.drop({'keyword', 'location'}, axis=1)
trainx = _input1['text']
trainy = _input1['target']
testx = _input0['text']
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=1.0))