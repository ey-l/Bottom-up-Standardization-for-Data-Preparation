import numpy as np
import pandas as pd
import nltk
import pydash
import math
import os
import time
from pydash import flatten
from collections import Counter, OrderedDict
from humanize import intcomma
from operator import itemgetter
from typing import *
from sklearn.model_selection import train_test_split
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col=0)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv', index_col=0)
_input1

def tokenize_df(dfs: List[pd.DataFrame], keys=('text', 'keyword', 'location'), stemmer=False, ngrams=1, preserve_case=True, reduce_len=False, strip_handles=True, use_stopwords=True, **kwargs) -> List[List[str]]:
    tokenizer = nltk.TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles)
    porter = nltk.PorterStemmer()
    stopwords = set(nltk.corpus.stopwords.words('english') + ['nan'])
    output = []
    for df in flatten([dfs]):
        for (index, row) in df.iterrows():
            tokens = flatten([tokenizer.tokenize(str(row[key] or '')) for key in keys])
            if use_stopwords:
                tokens = [token for token in tokens if token.lower() not in stopwords and len(token) >= 2]
            if stemmer:
                tokens = [porter.stem(token) for token in tokens]
            if ngrams:
                tokens = [' '.join(tokens[i:i + n]) for n in range(1, ngrams + 1) for i in range(0, len(tokens) - n + 1)]
            output.append(tokens)
    return output

def get_labeled_tokens(df, **kwargs) -> Dict[int, List[str]]:
    tokens = {0: flatten(tokenize_df(df[df['target'] == 0], **kwargs)), 1: flatten(tokenize_df(df[df['target'] == 1], **kwargs))}
    return tokens

def get_word_frequencies(df, **kwargs) -> Dict[int, Counter]:
    tokens = get_labeled_tokens(df, **kwargs)
    freqs = {0: Counter(dict(Counter(tokens[0]).most_common())), 1: Counter(dict(Counter(tokens[1]).most_common()))}
    return freqs

def get_log_likelihood(df, vocab_df, **kwargs):
    vocab = set(flatten(tokenize_df(vocab_df, **kwargs)))
    tokens = tokenize_df(df, **kwargs)
    freqs = get_word_frequencies(df, **kwargs)
    log_likelihood = {}
    for token in vocab:
        p_false = (freqs[0].get(token, 0) + 1) / (len(tokens[0]) + len(vocab))
        p_true = (freqs[1].get(token, 0) + 1) / (len(tokens[1]) + len(vocab))
        log_likelihood[token] = np.log(p_true / p_false)
    return log_likelihood

def get_logprior(df, **kwargs):
    """ Log probability of a word being positive given imbalanced data """
    tokens = tokenize_df(df, **kwargs)
    return np.log(len(tokens[0]) / len(tokens[1])) if len(tokens[1]) else 0
    return np.log(len(tokens[0]) / len(tokens[1])) if len(tokens[1]) else 0

def print_logprior():
    tokens = get_labeled_tokens(_input1)
    logprior = get_logprior(_input1)
    print('len(tokens[0])                    =', len(tokens[0]))
    print('len(tokens[1])                    =', len(tokens[1]))
    print('logprior(df_test)                 =', logprior)
    print('math.exp(logprior(df_test))       =', math.exp(logprior))
    print('math.exp(logprior(df_test)) ** -1 =', math.exp(logprior) ** (-1))
print_logprior()

def naive_bayes_classifier(df_train, df_test, **kwargs) -> np.array:
    vocab_df = [_input1, _input0]
    log_likelihood = get_log_likelihood(_input1, vocab_df, **kwargs)
    logprior = get_logprior(_input1, **kwargs)
    predictions = []
    for tweet_tokens in tokenize_df(_input0, **kwargs):
        log_prob = np.sum([log_likelihood.get(token, 0) for token in tweet_tokens]) + logprior
        prediction = int(log_prob > 0)
        predictions.append(prediction)
    return np.array(predictions)

def test_accuracy(splits=3, **kwargs):
    time_start = time.perf_counter()
    accuracy = 0
    for _ in range(splits):
        (train, test) = train_test_split(_input1, test_size=1 / splits)
        predictions = naive_bayes_classifier(train, test, **kwargs)
        accuracy += np.sum(test['target'] == predictions) / len(predictions) / splits
    time_taken = time.perf_counter() - time_start
    time_taken /= splits
    print(f'ngrams = {ngrams} | accuracy = {accuracy * 100:.2f}% | time = {time_taken:.1f}s')
for ngrams in [1, 2, 3, 4, 5]:
    test_accuracy(splits=3, ngrams=ngrams)
kwargs = {'ngrams': 3}
df_submission = pd.DataFrame({'id': _input0.index, 'target': naive_bayes_classifier(_input1, _input0, **kwargs)})