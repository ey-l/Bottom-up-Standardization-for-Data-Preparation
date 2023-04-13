import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def preprocess(data):
    porter = PorterStemmer()
    tweets = []
    rows = len(data)
    for index in range(rows):
        tweet = re.sub('[^\\w]', ' ', data[index])
        tweet = tweet.lower().split()
        tweet = [porter.stem(word) for word in tweet if word not in stopwords.words('english')]
        tweet = ' '.join(tweet)
        tweets.append(tweet)
    return tweets
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1.info()
_input1.isna().sum()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
tweets = preprocess(_input1['text'])
X = cv.fit_transform(tweets).toarray()
X.shape
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, _input1['target'], test_size=0.33, random_state=35)