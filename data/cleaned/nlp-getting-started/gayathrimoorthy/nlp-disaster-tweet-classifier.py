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
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
train.info()
train.isna().sum()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
tweets = preprocess(train['text'])
X = cv.fit_transform(tweets).toarray()
X.shape
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, train['target'], test_size=0.33, random_state=35)