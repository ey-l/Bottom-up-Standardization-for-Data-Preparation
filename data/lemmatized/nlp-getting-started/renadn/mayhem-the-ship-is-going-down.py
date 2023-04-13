import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1.target.value_counts()
_input0.head()
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    Note: 
        I have taken the liberty of altering and subsequently utilizing this function from DeepLearning.AI 
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub('\\$\\w*', '', tweet)
    tweet = re.sub('^RT[\\s]+', '', tweet)
    tweet = re.sub('https?:\\/\\/.*[\\r\\n]*', '', tweet)
    tweet = re.sub('#', '', tweet)
    tweet = ' '.join((word for word in tweet.split(' ') if word not in stopwords_english))
    tweet = ' '.join((stemmer.stem(word) for word in tweet.split(' ')))
    return tweet
print('This is an example of a positive tweet: \n', _input1.iloc[1]['text'])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(_input1.iloc[1]['text']))
_input1['processed'] = _input1['text'].apply(process_tweet)
_input0['processed'] = _input0['text'].apply(process_tweet)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
X = _input1.processed
y = _input1['target']
X_test = _input0.processed
(X_train, X_val, y_train, y_val) = train_test_split(X, y, random_state=42, test_size=0.2)
vectorizer = CountVectorizer()