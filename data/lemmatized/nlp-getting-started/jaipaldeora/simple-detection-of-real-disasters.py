import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col='id')
_input1.head()
_input1.info()
sns.catplot(kind='count', data=_input1, x='target', aspect=3)
_input1 = _input1.drop(['location', 'keyword'], inplace=False, axis=1)
_input1.head

def process_tweet(tweet):
    stopwords_english = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tweet = re.sub('\\$\\w*', '', tweet)
    tweet = re.sub('^RT[\\s]+', '', tweet)
    tweet = re.sub('https?:\\/\\/.*[\\r\\n]*', '', tweet)
    tweet = re.sub('#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    clean = []
    for word in tokens:
        if word not in stopwords_english and word not in string.punctuation:
            stem_word = stemmer.stem(word)
            clean.append(stem_word)
    return clean

def build_freqs(tweets, targets):
    yslist = np.squeeze(targets).tolist()
    freqs = {}
    for (target, tweet) in zip(targets, tweets):
        for word in process_tweet(tweet):
            pair = (word, target)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs
X = _input1['text'].values
y = _input1['target'].values
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.1, stratify=y, random_state=43)
freqs = build_freqs(X_train, y_train)

def extract_features(tweet, freqs):
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in word_l:
        x[0, 1] += freqs.get((word, 1), 0)
        x[0, 2] += freqs.get((word, 0), 0)
    return x

def train(x, y, xVal, yVal, freqs):
    xTRAIN = np.zeros((len(x), 3))
    for i in range(len(x)):
        xTRAIN[i, :] = extract_features(x[i], freqs)
    model = SGDClassifier(loss='log', n_jobs=-1, max_iter=800, random_state=31)