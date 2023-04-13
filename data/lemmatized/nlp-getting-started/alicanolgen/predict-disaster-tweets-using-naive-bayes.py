import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.info()
x_train = _input1['text']
y_train = _input1['target']
'have given needed library, module'
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
import string

def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub('\\$\\w*', '', tweet)
    tweet = re.sub('^RT[\\s]+', '', tweet)
    tweet = re.sub('https?:\\/\\/.*[\\r\\n]*', '', tweet)
    tweet = re.sub('#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean
'example for created process_tweet() func.'
custom_tweet = 'OMG house burned :( #bad #morning http://fire.com'
print(process_tweet(custom_tweet))

def lookup(freqs, word, label):
    n = 0
    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]
    return n

def count_tweets(result, tweets, ys):
    for (y, tweet) in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result
freqs = count_tweets({}, x_train, y_train)

def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    N_real = N_notreal = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_real += freqs[pair]
        else:
            N_notreal += freqs[pair]
    D = len(train_x)
    D_real = sum(train_y == 1)
    D_notreal = sum(train_y == 0)
    logprior = np.log(D_real) - np.log(D_notreal)
    for word in vocab:
        freq_real = lookup(freqs, word, 1)
        freq_notreal = lookup(freqs, word, 0)
        p_w_real = (freq_real + 1) / (N_real + V)
        p_w_notreal = (freq_notreal + 1) / (N_notreal + V)
        loglikelihood[word] = np.log(p_w_real) - np.log(p_w_notreal)
    return (logprior, loglikelihood)
(logprior, loglikelihood) = train_naive_bayes(freqs, x_train, y_train)
print(logprior)
print(len(loglikelihood))

def naive_bayes_predict(tweet, logprior, loglikelihood):
    word_l = process_tweet(tweet)
    p = 0
    p += logprior
    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p
my_tweet = '@sakuma_en If you pretend to feel a certain way the feeling can become genuine all by accident. -Hei...'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0
    y_hats = []
    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)
    error = np.mean(np.absolute(y_hats - test_y))
    accuracy = 1 - error
    return accuracy
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
y_predicted = []
for data in _input0['text']:
    result = naive_bayes_predict(data, logprior, loglikelihood)
    if result >= 0:
        y_predicted.append(1)
    if result < 0:
        y_predicted.append(0)
y_predicted
iyd = _input0['id']
trgt = y_predicted
dict = {'id': iyd, 'target': trgt}
df = pd.DataFrame(dict)
df.head()