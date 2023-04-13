import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
print('Number of positive tweets ', len(_input1[_input1['target'] == 0]))
print('Number of negative tweets ', len(_input1[_input1['target'] == 1]))
import nltk
import string
import re

def preprocess(tweet):
    tweet = re.sub('^RT[\\s]+', '', tweet)
    tweet = re.sub('https?:\\/\\/.*[\\r\\n]*', '', tweet)
    tweet = re.sub('#', '', tweet)
    tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokenized_tweet = tokenizer.tokenize(tweet)
    stemmer = nltk.stem.PorterStemmer()
    english_stopwords = nltk.corpus.stopwords.words('english')
    processed_tweet = []
    for word in tokenized_tweet:
        if word not in english_stopwords and word not in string.punctuation:
            processed_tweet.append(stemmer.stem(word))
    return processed_tweet
print('Original Tweet: ', _input1['text'][5])
print('Processed Tweet: ', preprocess(_input1['text'][5]))

def freq_builder(tweets, labels):
    freq_dict = {}
    labels_list = np.squeeze(labels).tolist()
    for (tweet, l) in zip(tweets, labels_list):
        for word in preprocess(tweet):
            pair = (word, l)
            if pair in freq_dict:
                freq_dict[pair] += 1
            else:
                freq_dict[pair] = 1
    return freq_dict
frequency = freq_builder(_input1['text'], _input1['target'])
print(list(frequency.items())[5:10])

def feature_extraction(tweet, freq_dict):
    tokenized_tweet = preprocess(tweet)
    f = np.zeros((1, 3))
    f[0, 0] = 1
    for word in tokenized_tweet:
        f[0, 1] += freq_dict.get((word, 1.0), 0)
        f[0, 2] += freq_dict.get((word, 0.0), 0)
    return f
X_train = np.zeros((len(_input1['target']), 3))
for i in range(len(_input1['target'])):
    X_train[i, :] = feature_extraction(_input1['text'][i], frequency)
Y_train = _input1['target']
print(X_train[10:15])
print(Y_train[10:15])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
svm = SVC(C=1.0, kernel='rbf')
result = cross_validate(svm, X_train, Y_train, scoring=('accuracy', 'f1'), cv=3, return_train_score=True)
print('Train Accuracy Score: ', result['train_accuracy'])
print('Test Accuracy Score: ', result['test_accuracy'])
print('Train F1 Score: ', result['train_f1'])
print('Test F1 Score: ', result['test_f1'])
X_test = np.zeros((len(_input0['text']), 3))