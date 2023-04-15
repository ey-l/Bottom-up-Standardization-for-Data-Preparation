import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
data.head()
(data.shape, test.shape)
data.isna().sum()
data = data.drop(['location'], axis=1)
data.fillna(data.median(), inplace=True)
data = data.drop(['keyword'], axis=1)
data.target.value_counts().plot(kind='bar', color='red')

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
data['text'] = data['text'].apply(newtweet)
test['text'] = test['text'].apply(newtweet)
test = test.drop({'keyword', 'location'}, axis=1)
trainx = data['text']
trainy = data['target']
testx = test['text']
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=1.0))