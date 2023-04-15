import numpy as np
import pandas as pd
import nltk
import pydash
import math
import os
from pydash import flatten
from collections import Counter, OrderedDict
from humanize import intcomma
from operator import itemgetter
from typing import *
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col=0)
df_test = pd.read_csv('data/input/nlp-getting-started/test.csv', index_col=0)
df_train

def tokenize_df(dfs: List[pd.DataFrame]) -> List[str]:
    tokenizer = nltk.TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
    tokens = flatten([tokenizer.tokenize(tweet_text) for df in flatten([dfs]) for tweet_text in df['text']])
    return tokens
tokens_all = tokenize_df([df_train, df_test])
tokens_train = tokenize_df(df_train)
tokens_test = tokenize_df(df_test)
tokens_disaster = tokenize_df(df_train[df_train['target'] == 1])
tokens_not_disaster = tokenize_df(df_train[df_train['target'] == 0])
tokens_shared = set(tokens_train) & set(tokens_test) & set(tokens_disaster) & set(tokens_not_disaster)
print('Unique Tokens:')
print('  tokens_all          ', intcomma(len(set(tokens_all))))
print('  tokens_train        ', intcomma(len(set(tokens_train))))
print('  tokens_test         ', intcomma(len(set(tokens_test))))
print('  tokens_disaster     ', intcomma(len(set(tokens_disaster))))
print('  tokens_not_disaster ', intcomma(len(set(tokens_not_disaster))))
print('  tokens_shared       ', intcomma(len(set(tokens_shared))))
print()
print('New Tokens:')
print(f'  tokens_test         - tokens_train        {intcomma(len(set(tokens_test) - set(tokens_train))):>6s} ({len(set(tokens_test) - set(tokens_train)) / len(set(tokens_test)) * 100:.1f}%)')
print(f'  tokens_train        - tokens_test         {intcomma(len(set(tokens_train) - set(tokens_test))):>6s} ({len(set(tokens_train) - set(tokens_test)) / len(set(tokens_train)) * 100:.1f}%)')
print(f'  tokens_disaster     - tokens_not_disaster {intcomma(len(set(tokens_disaster) - set(tokens_not_disaster))):>6s} ({len(set(tokens_disaster) - set(tokens_not_disaster)) / len(set(tokens_disaster)) * 100:.1f}%)')
print(f'  tokens_not_disaster - tokens_disaster     {intcomma(len(set(tokens_not_disaster) - set(tokens_disaster))):>6s} ({len(set(tokens_not_disaster) - set(tokens_disaster)) / len(set(tokens_not_disaster)) * 100:.1f}%)')

def term_frequency(tokens: List[str]) -> Counter:
    tf = {token: count / len(tokens) for (token, count) in Counter(tokens).items()}
    tf = Counter(dict(Counter(tf).most_common()))
    return tf

def inverse_document_frequency(tokens: List[str]) -> Counter:
    idf = {token: math.log(len(tokens) / count) for (token, count) in Counter(tokens).items()}
    idf = Counter(dict(Counter(idf).most_common()))
    return idf

def tf_idf(document_tokens: List[str], all_tokens: List[str]) -> Counter:
    tf = term_frequency(document_tokens)
    idf = inverse_document_frequency(all_tokens)
    tf_idf = {token: tf[token] * idf[token] for token in set(document_tokens)}
    tf_idf = Counter(dict(Counter(tf_idf).most_common()))
    return tf_idf
tf_disaster = term_frequency(tokens_disaster)
tf_not_disaster = term_frequency(tokens_not_disaster)
idf = inverse_document_frequency(tokens_all)
tf_idf_disaster = tf_idf(tokens_disaster, tokens_all)
tf_idf_not_disaster = tf_idf(tokens_not_disaster, tokens_all)









def tf_idf_ratio(tf_idf_true: Counter, tf_idf_false: Counter) -> Counter:
    tf_idf_false_tokens = set(tf_idf_false.keys())
    tf_idf_ratio = {token: tf_idf_true[token] / tf_idf_false[token] for token in tf_idf_true.keys() if token in tf_idf_false_tokens}
    tf_idf_ratio = Counter(dict(Counter(tf_idf_ratio).most_common()))
    return tf_idf_ratio
tf_idf_ratio_disaster = tf_idf_ratio(tf_disaster, tf_not_disaster)
tf_idf_ratio_not_disaster = tf_idf_ratio(tf_not_disaster, tf_disaster)





def tf_idf_classifer_score(tweet_text: str, tf_idf_ratio_disaster, tf_idf_ratio_not_disaster) -> float:
    score = 0.0
    tokens = nltk.TweetTokenizer().tokenize(tweet_text)
    for token in tokens:
        if token in tokens_shared:
            score += math.log(tf_idf_ratio_disaster.get(token, 1))
            score -= math.log(tf_idf_ratio_not_disaster.get(token, 1))
    return score

def tf_idf_classifer(tweet_text: str, tf_idf_ratio_disaster, tf_idf_ratio_not_disaster) -> int:
    score = tf_idf_classifer_score(tweet_text, tf_idf_ratio_disaster, tf_idf_ratio_not_disaster)
    label = 1 if score > 0 else 0
    return label

def tf_idf_classifer_df(df: pd.DataFrame) -> np.ndarray:
    return np.array([tf_idf_classifer(row['text'], tf_idf_ratio_disaster, tf_idf_ratio_not_disaster) for (index, row) in df.iterrows()])

def tf_idf_classifer_accuracy(df: pd.DataFrame, tf_idf_ratio_disaster, tf_idf_ratio_not_disaster) -> float:
    correct = 0
    total = 0
    for (index, row) in df.iterrows():
        label = tf_idf_classifer(row['text'], tf_idf_ratio_disaster, tf_idf_ratio_not_disaster)
        if label == row['target']:
            correct += 1
        total += 1
    accuracy = correct / total
    return accuracy
accuracy = tf_idf_classifer_accuracy(df_train, tf_idf_ratio_disaster, tf_idf_ratio_not_disaster)
print('accuracy =', accuracy)
df_train
predictions = tf_idf_classifer_df(df_train)
tokens_true_positive = tokenize_df(df_train[df_train['target'] == predictions])
tokens_false_positive = tokenize_df(df_train[df_train['target'] != predictions])
tf_idf_true_positive = tf_idf(tokens_true_positive, tokens_all)
tf_idf_false_positive = tf_idf(tokens_false_positive, tokens_all)
tf_idf_true_positive_ratio = tf_idf_ratio(tf_idf_true_positive, tf_idf_false_positive)
tf_idf_false_positive_ratio = tf_idf_ratio(tf_idf_false_positive, tf_idf_true_positive)





def ratio_hyperparameter_search():
    results = {}
    for scale1 in [0.5, 1, 1.5, 2, 2.5, 3, 4, 8, 16]:
        for scale2 in [0.5, 1, 1.5, 2, 2.5, 3, 4, 8, 16]:
            tf_idf_true_positive_ratio_scaled = Counter({token: count / scale1 for (token, count) in tf_idf_true_positive_ratio.items()})
            tf_idf_false_positive_ratio_scaled = Counter({token: count / scale2 for (token, count) in tf_idf_false_positive_ratio.items()})
            accuracy = tf_idf_classifer_accuracy(df_train, tf_idf_ratio_disaster + tf_idf_true_positive_ratio_scaled, tf_idf_ratio_not_disaster + tf_idf_false_positive_ratio_scaled)
            results[scale1, scale2] = accuracy

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost') == 'Batch':
    ratio_hyperparameter_search()
tf_idf_true_positive_ratio_scaled = Counter({token: count / 3 for (token, count) in tf_idf_true_positive_ratio.items()})
tf_idf_false_positive_ratio_scaled = Counter({token: count / 2 for (token, count) in tf_idf_false_positive_ratio.items()})
accuracy = tf_idf_classifer_accuracy(df_train, tf_idf_ratio_disaster + tf_idf_true_positive_ratio_scaled, tf_idf_ratio_not_disaster + tf_idf_false_positive_ratio_scaled)
print('accuracy =', accuracy)
df_submission = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv', index_col=0)
for (index, row) in df_test.iterrows():
    label = tf_idf_classifer(row['text'], tf_idf_ratio_disaster + tf_idf_true_positive_ratio_scaled, tf_idf_ratio_not_disaster + tf_idf_false_positive_ratio_scaled)
    df_submission.loc[index] = label

df_submission