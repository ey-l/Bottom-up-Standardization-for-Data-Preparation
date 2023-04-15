import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
KAGGLE_HOME = 'data/input/nlp-getting-started'
tweets = pd.read_csv(os.path.join(KAGGLE_HOME, 'train.csv'))
tweets.head()
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
URL_PATTERN = '((http|https)\\:\\/\\/)?[a-zA-Z0-9\\.\\/\\?\\:@\\-_=#]+\\.([a-zA-Z]){2,6}([a-zA-Z0-9\\.\\&\\/\\?\\:@\\-_=#])*'
all_stopwords = stopwords.words('english')

def process_text(text):
    remove_stop = ' '.join([word for word in text.split() if word not in all_stopwords])
    remove_url = re.sub(URL_PATTERN, '', remove_stop)
    remove_punc = re.sub('[^\\w\\s]', '', remove_url)
    return remove_punc.lower()
tweets['processed_text'] = tweets['text'].apply(lambda text: process_text(text))
print(f'\nBefore text processing: \n{tweets.text[100]}')
print(f'\nAfter text processing: \n{process_text(tweets.text[100])}')
from collections import defaultdict

def create_word_index(string):
    freq_dict = defaultdict(int)
    for word in string.split():
        if word not in freq_dict:
            freq_dict[word] = 1
        else:
            freq_dict[word] += 1
    return freq_dict
positive_corpus = ' '.join((text for text in tweets[tweets['target'] == 1]['processed_text']))
negative_corpus = ' '.join((text for text in tweets[tweets['target'] == 0]['processed_text']))
pos_freq_dict = create_word_index(positive_corpus)
neg_freq_dict = create_word_index(negative_corpus)

def extract_features(freq_dict, tweet):
    freq = 0
    for word in tweet.split():
        freq += freq_dict[word]
    return freq
tweets['pos_freq'] = tweets['processed_text'].apply(lambda tweet: extract_features(pos_freq_dict, tweet))
tweets['neg_freq'] = tweets['processed_text'].apply(lambda tweet: extract_features(neg_freq_dict, tweet))
tweets.head()

def split_train_test(features, labels, split_size):
    train_size = int(len(features) * split_size)
    data = list(zip(features, labels))
    shuffle_data = random.sample(data, len(data))
    shuffle_features = [feature for (feature, label) in shuffle_data]
    shuffle_labels = [label for (feature, label) in shuffle_data]
    x_train = np.array(shuffle_features[:train_size])
    y_train = np.array(shuffle_labels[:train_size]).reshape((len(shuffle_labels[:train_size]), 1))
    x_test = np.array(shuffle_features[train_size:])
    y_test = np.array(shuffle_labels[train_size:]).reshape((len(shuffle_labels[train_size:]), 1))
    return (x_train, x_test, y_train, y_test)
import seaborn as sns
import matplotlib.pyplot as plt
tweets['norm_pos_freq'] = (tweets['pos_freq'] - tweets['pos_freq'].mean()) / tweets['pos_freq'].std()
tweets['norm_neg_freq'] = (tweets['neg_freq'] - tweets['neg_freq'].mean()) / tweets['neg_freq'].std()
(fig, axes) = plt.subplots(ncols=2, figsize=(15, 5))
sns.scatterplot(x='pos_freq', y='neg_freq', hue='target', data=tweets, ax=axes[0], alpha=0.1)
sns.scatterplot(x='norm_pos_freq', y='norm_neg_freq', hue='target', data=tweets, ax=axes[1], alpha=0.1)

class LogisticRegression:

    def __init__(self):
        weight = None
        costs = []
        accuracies = []

    def sigmoid(self, x):
        output = 1 / (1 + np.exp(-x))
        return output

    def compute_cost(self, y_pred, y):
        error = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        return -np.mean(error)

    def compute_gradient(self, X, y, y_pred):
        gradient = 1 / len(X) * np.dot(X.T, y_pred - y)
        return gradient

    def fit(self, X, y, epoch, learning_rate):
        self.weight = np.zeros((X.shape[1], 1))
        costs = []
        accuracies = []
        for _ in tqdm(range(epoch)):
            y_pred = self.sigmoid(np.dot(X, self.weight))
            cost = self.compute_cost(y_pred, y)
            gradient = self.compute_gradient(X, y, y_pred)
            self.weight -= learning_rate * gradient
            accuracy = self.score(X, y)
            costs.append(cost)
            accuracies.append(accuracy)
        self.costs = costs
        self.accuracies = accuracies

    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.weight))
        return y_pred > 0.5

    def predict_prob(self, X):
        y_prob = self.sigmoid(np.dot(X, self.weight))
        return y_prob

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
EPOCH = 500
LEARNING_RATE = 1
tweets_text = tweets[['pos_freq', 'neg_freq']].values
mean = np.mean(tweets_text, axis=0)
std = np.std(tweets_text, axis=0)
tweets_text = (tweets_text - mean) / std
labels = tweets.copy().pop('target').values
(X_train, X_test, y_train, y_test) = split_train_test(tweets_text, labels, 0.8)
X_train = np.append(np.ones((X_train.shape[0], 1)), X_train, axis=1)
X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)
log_reg = LogisticRegression()