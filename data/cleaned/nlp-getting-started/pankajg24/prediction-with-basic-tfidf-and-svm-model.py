import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import nltk
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('data/input/nlp-getting-started/train.csv')
data.head()

def preprocess(tweets):
    tweets_lower = [tweet.lower() for tweet in tweets]
    tweets_re = [re.sub('http\\S+', '', tweet) for tweet in tweets_lower]
    stop_words = stopwords.words('english')
    tweet_token = []
    ps = PorterStemmer()
    clean_tweet = []
    for tweet in tweets_re:
        words = nltk.word_tokenize(tweet)
        tweet_token = [ps.stem(word) for word in words if word not in stop_words and word not in string.punctuation]
        tweet_sent = ' '.join(tweet_token)
        clean_tweet.append(tweet_sent)
    return clean_tweet
print(preprocess(data.text[:3]))
temp = data[data.id == 445]
print(preprocess(temp.text))
temp.columns
data_clean = preprocess(data.text)
target = data.target
data_clean[:5]
(x_train, x_test, y_train, y_test) = train_test_split(data_clean, target, test_size=0.2, stratify=target, random_state=123)
x_train[:5]