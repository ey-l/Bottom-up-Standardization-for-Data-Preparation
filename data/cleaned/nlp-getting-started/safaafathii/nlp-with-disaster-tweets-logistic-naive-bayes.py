import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
import nltk
import matplotlib.pyplot as plt
import random
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col='id')
df_train.head()
df_train.shape
df_train[df_train['target'] == 1].shape
df_train[df_train['target'] == 0].shape
fig = plt.figure(figsize=(5, 5))
labels = ('Disaster', 'Non-Disaster')
sizes = [df_train[df_train['target'] == 1].shape[0], df_train[df_train['target'] == 0].shape[0]]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')

df_dis = df_train[df_train['target'] == 1]
df_dis.shape
df_non_dis = df_train[df_train['target'] == 0]
df_non_dis.shape
print('\x1b[92m' + df_non_dis.iloc[random.randint(0, 4342), 2])
print('\x1b[91m' + df_dis.iloc[random.randint(0, 3271), 2])
tweet = df_dis.iloc[random.randint(0, 3271), 2]
tweet
tweet2 = re.sub('^RT[\\s]+', '', tweet)
tweet2 = re.sub('https?:\\/\\/.*[\\r\\n]*', '', tweet2)
tweet2 = re.sub('#', '', tweet2)
print('\x1b[92m' + tweet)
print('\x1b[94m')
print(tweet2)
print('\x1b[92m' + tweet2)
print('\x1b[94m')
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet2)
print()
print('Tokenized string:')
print(tweet_tokens)
stopwords_english = stopwords.words('english')
print('Stop words\n')
print(stopwords_english)
print('\nPunctuation\n')
print(string.punctuation)
print()
print('\x1b[92m')
print(tweet_tokens)
print('\x1b[94m')
tweets_clean = []
for word in tweet_tokens:
    if word not in stopwords_english and word not in string.punctuation:
        tweets_clean.append(word)
print('removed stop words and punctuation:')
print(tweets_clean)
print()
print('\x1b[92m')
print(tweets_clean)
print('\x1b[94m')
stemmer = PorterStemmer()
tweets_stem = []
for word in tweets_clean:
    stem_word = stemmer.stem(word)
    tweets_stem.append(stem_word)
print('stemmed words:')
print(tweets_stem)

def preprocessing(df):
    try:
        df['text'] = df.loc[:, 'text'].apply(lambda x: re.sub('^RT[\\s]+', '', x))
        df['text'] = df.loc[:, 'text'].apply(lambda x: re.sub('https?:\\/\\/.*[\\r\\n]*', '', x))
        df['text'] = df.loc[:, 'text'].apply(lambda x: re.sub('#', '', x))
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        df['text'] = df.loc[:, 'text'].apply(lambda x: tokenizer.tokenize(x))
        stopwords_english = stopwords.words('english')

        def remove_Stop_Punc(listt):
            clean = []
            for word in listt:
                if word not in stopwords_english and word not in string.punctuation:
                    clean.append(word)
            return clean
        df['text'] = df.loc[:, 'text'].apply(lambda x: remove_Stop_Punc(x))
        stemmer = PorterStemmer()

        def stemming_func(listt):
            stemmed = []
            for word in listt:
                stem_word = stemmer.stem(word)
                stemmed.append(stem_word)
            return stemmed
        df['text'] = df.loc[:, 'text'].apply(lambda x: stemming_func(x))
    except:
        print('Already Preprocessed')
preprocessing(df_train)
print('------------------------------------------------------------')
df_train.head()
wordFreq = {}
for (index, row) in df_train.iterrows():
    for word in row['text']:
        pair = (word, row['target'])
        if pair in wordFreq:
            wordFreq[pair] += 1
        else:
            wordFreq[pair] = 1
wordFreq
Xm = []
for (index, row) in df_train.iterrows():
    tweetFeatureList = []
    posFreq = 0
    negFreq = 0
    for word in row['text']:
        if (word, 1) in wordFreq:
            posFreq += wordFreq[word, 1]
        if (word, 0) in wordFreq:
            negFreq += wordFreq[word, 0]
    tweetFeatureList.append(1)
    tweetFeatureList.append(posFreq)
    tweetFeatureList.append(negFreq)
    Xm.append(tweetFeatureList)
len(Xm)
Xm
data = pd.DataFrame(Xm, columns=['Bias', 'PosFreq', 'NegFreq'])
data.head()
df_train.reset_index(drop=True, inplace=True)
data['Sentiment'] = df_train['target']
data.head()
X = data.drop('Sentiment', axis=1)
y = data['Sentiment']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy(dtype='float64')
y_train = np.reshape(y_train, (y_train.shape[0], 1))
m = X_train.shape[0]
alpha = 1e-08
theta = np.zeros((3, 1))
for i in range(0, 1500):
    z = np.dot(X_train, theta)
    Ypred = 1 / (1 + np.exp(-z))
    cost = -1 / m * (np.dot(y_train.T, np.log(Ypred)) + np.dot((1 - y_train).T, np.log(1 - Ypred)))
    theta = theta - alpha / m * np.dot(X_train.T, Ypred - y_train)
theta
y_test = y_test.to_numpy(dtype='float64')
y_test = np.reshape(y_test, (y_test.shape[0], 1))
ypredicted = []
ztest = np.dot(X_test, theta)
Ytest = 1 / (1 + np.exp(-ztest))
for i in Ytest:
    if i > 0.5:
        ypredicted.append(1.0)
    else:
        ypredicted.append(0.0)
accuracy = (ypredicted == np.squeeze(y_test)).sum() / len(X_test)
accuracy
wordFreq
X = df_train.drop('target', axis=1)
y = df_train['target']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
ProbPos = y_train[y_train == 1].shape[0] / y_train.shape[0]
ProbPos
ProbNeg = y_train[y_train == 0].shape[0] / y_train.shape[0]
ProbNeg
import math
logPrior = math.log(ProbPos / ProbNeg)
logPrior
Xm = []
V = set([pair[0] for pair in wordFreq.keys()])
NPos = NNeg = 0
for (index, row) in X_train.iterrows():
    for word in row['text']:
        if (word, 1) in wordFreq:
            NPos += wordFreq[word, 1]
        if (word, 0) in wordFreq:
            NNeg += wordFreq[word, 0]
len(V)
likelihoodMatrix = {}
for (index, row) in X_train.iterrows():
    ProbWPos = 0
    ProbWNeg = 0
    for word in row['text']:
        if (word, 1) in wordFreq:
            ProbWPos = (wordFreq[word, 1] + 1) / (NPos + len(V))
        else:
            ProbWPos = (0 + 1) / (NPos + len(V))
        if (word, 0) in wordFreq:
            ProbWNeg = (wordFreq[word, 0] + 1) / (NNeg + len(V))
        else:
            ProbWNeg = (0 + 1) / (NNeg + len(V))
        LogLikelihood = math.log(ProbWPos / ProbWNeg)
        likelihoodMatrix[word] = LogLikelihood
likelihoodMatrix
len(likelihoodMatrix)
YTrainPred = []
for tweet in X_train['text']:
    p = 0
    for word in tweet:
        p += logPrior + likelihoodMatrix[word]
    if p > 0:
        YTrainPred.append(1)
    else:
        YTrainPred.append(0)
error = np.mean(np.absolute(YTrainPred - y_train))
accuracy = 1 - error
accuracy