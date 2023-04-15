import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
tweet_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
tweet_df.head()
tweet_df.target.value_counts()
test_df.head()
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
print('This is an example of a positive tweet: \n', tweet_df.iloc[1]['text'])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(tweet_df.iloc[1]['text']))
tweet_df['processed'] = tweet_df['text'].apply(process_tweet)
test_df['processed'] = test_df['text'].apply(process_tweet)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
X = tweet_df.processed
y = tweet_df['target']
X_test = test_df.processed
(X_train, X_val, y_train, y_val) = train_test_split(X, y, random_state=42, test_size=0.2)
vectorizer = CountVectorizer()