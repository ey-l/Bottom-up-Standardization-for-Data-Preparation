
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sbn
import sys
import os
import re
import string
import scipy
from scipy.stats import chi2_contingency
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import fdrcorrection, multipletests
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import contractions
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
print('python', sys.version)
for module in (np, pd, mpl, sbn, nltk, sk, nltk, re, scipy):
    print(module.__name__, module.__version__)
np.random.seed(0)
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
sample_submission = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
pd.set_option('display.max_colwidth', None)
train.head()
test.head()
print('The training set has {} rows and {} columns.'.format(train.shape[0], train.shape[1]))
print('The test set has {} rows and {} columns.'.format(test.shape[0], test.shape[1]))
print('The training set has {} duplicated rows.'.format(train.drop('id', axis=1).duplicated(keep=False).sum()))
print('The test set has {} duplicated rows.'.format(test.drop('id', axis=1).duplicated(keep=False).sum()))
train[train.drop(['id', 'target'], axis=1).duplicated(keep=False) & ~train.drop(['id'], axis=1).duplicated(keep=False)]
train[train.text.str.contains('CLEARED:incident with injury:I-495')]
test[test.drop(['id'], axis=1).duplicated(keep=False)].iloc[:10, :]
train[train['text'].duplicated(keep=False) & ~train.drop(['id', 'target'], axis=1).duplicated(keep=False)].sort_values(by='text')[10:20]
print('There are {} such tweets.'.format(train[train['text'].duplicated(keep=False) & ~train.drop(['id', 'target'], axis=1).duplicated(keep=False)].shape[0]))
print('{}% of the locations are missing in the training set and {}% in the test set'.format(round(train.location.isnull().sum() / train.shape[0] * 100, 1), round(test.location.isnull().sum() / test.shape[0] * 100, 1)))
print('{}% of the keywords are missing in the training set and {}% in the test set'.format(round(train.keyword.isnull().sum() / train.shape[0] * 100, 2), round(test.keyword.isnull().sum() / test.shape[0] * 100, 2)))
zeros = round((train.target == 0).sum() / train.shape[0], 2)
ones = round(1 - zeros, 2)
sbn.barplot(x=['Non disasters', 'Disasters'], y=[zeros, ones], color='gray')
plt.gca().set_ybound(0, 0.7)
plt.gca().set_ylabel('Proportion of tweets')
plt.gca().set_yticklabels([])
plt.gca().tick_params(axis='x')
plt.annotate(str(zeros) + '%', xy=(-0.1, zeros + 0.01), size=15)
plt.annotate(str(ones) + '%', xy=(0.9, ones + 0.01), size=15)
plt.suptitle('Distribution of disasters', size=15)

remove_url = lambda x: re.sub('https?:\\/\\/t.co\\/[A-Za-z0-9]+', '', x)
train['original_text'] = train.text.copy()
train['text'] = train.text.apply(remove_url)
test['original_text'] = test.text.copy()
test['text'] = test.text.apply(remove_url)
train.loc[train.text != train.original_text, ['original_text', 'text']].head()

def cleaning(data):
    data = data.apply(lambda x: re.sub('\\x89Ûª', "'", x))
    data = data.apply(lambda x: re.sub('&;amp;', '&', x))
    data = data.apply(lambda x: re.sub('&amp;', '&', x))
    data = data.apply(lambda x: re.sub('&amp', '&', x))
    data = data.apply(lambda x: re.sub('Û¢åÊ', '', x))
    data = data.apply(lambda x: re.sub('ÛÒåÊ', '', x))
    data = data.apply(lambda x: re.sub('Û_', '', x))
    data = data.apply(lambda x: re.sub('ÛÒ', '', x))
    data = data.apply(lambda x: re.sub('ÛÓ', '', x))
    data = data.apply(lambda x: re.sub('ÛÏ', '', x))
    data = data.apply(lambda x: re.sub('Û÷', '', x))
    data = data.apply(lambda x: re.sub('Ûª', '', x))
    data = data.apply(lambda x: re.sub('\\x89Û\\x9d', '', x))
    data = data.apply(lambda x: re.sub('Û¢', '', x))
    data = data.apply(lambda x: re.sub('åÈ', '', x))
    data = data.apply(lambda x: re.sub('åÊ', ' ', x))
    data = data.apply(lambda x: re.sub('å¨', '', x))
    data = data.apply(lambda x: re.sub('åÇ', '', x))
    data = data.apply(lambda x: re.sub('å_', '', x))
    return data
train.text = cleaning(train.text)
test.text = cleaning(test.text)
train.loc[train.original_text.str.contains('&amp'), ['original_text', 'text']].head()
vocabulary = set(train['text'].apply(lambda x: re.sub('[0-9]', '', x)).apply(lambda x: x.split()).sum())
vocabulary_test = set(test['text'].apply(lambda x: re.sub('[0-9]', '', x)).apply(lambda x: x.split()).sum())
print('The tweets from the traing set contain {} unique words (after removing the URL and the figures).'.format(len(vocabulary)))
print('The tweets from the test set contain {} unique words (after removing the URL and the figures).'.format(len(vocabulary_test)))
contractions_detected = pd.DataFrame({word: [contractions.fix(word)] for word in vocabulary if word != contractions.fix(word)}, index=['Corrections']).T
print('We detected {} differents contractions in the tweets from the traing set.'.format(contractions_detected.shape[0]))
contractions_detected[:10]

def check_contractions(w):
    if w in contractions_detected.index:
        return contractions_detected.loc[w, 'Corrections']
    else:
        return w
train.text = train.text.apply(lambda x: ' '.join([check_contractions(w) for w in x.split()]))
test.text = test.text.apply(lambda x: ' '.join([check_contractions(w) for w in x.split()]))
train.loc[train.original_text.str.contains('theres'), ['original_text', 'text']]
test.loc[test.original_text.str.contains("what's"), ['original_text', 'text']].head()
train.loc[~train.keyword.isna(), 'keyword'] = train.loc[~train.keyword.isna(), 'keyword'].apply(lambda x: re.sub('%20', ' ', str(x)))
test.loc[~test.keyword.isna(), 'keyword'] = test.loc[~test.keyword.isna(), 'keyword'].apply(lambda x: re.sub('%20', ' ', str(x)))
print('The training set contains {} keywords. The rarest appears in {} tweets and the least rare in {} tweets. The keyword variable also contains {} missing values.'.format(train.keyword.value_counts().shape[0], train.keyword.value_counts().min(), train.keyword.value_counts().max(), train.keyword.isna().sum()))
groupby_keyword = train[['keyword', 'target']].groupby('keyword')['target'].agg(frequencies='mean', count='size').reset_index().sort_values(by='count', ascending=False)
sbn.barplot(y='keyword', x='count', data=groupby_keyword.iloc[:20], color='gray')
plt.gca().set_xlabel('Count')
plt.gca().set_ylabel('Keywords')
plt.suptitle('20 most occurring Keywords', size=15)

groupby_keyword.sort_values(by='frequencies', ascending=False, inplace=True)
sbn.barplot(y='keyword', x='frequencies', data=groupby_keyword.iloc[:20], color='gray')
plt.gca().set_xlabel('Frequency of disasters')
plt.gca().set_ylabel('Keywords')
plt.suptitle('20 highest frequencies of disaster by keyword', size=15)

print('The 20 keywords associated with the highest frequencies of disaster occur between {} and {} times with an average of {}.'.format(groupby_keyword[:20]['count'].min(), groupby_keyword[:20]['count'].max(), groupby_keyword[:20]['count'].mean()))
print("{} samples have 'debris', 'wreckage' or 'derailment' for keyword.".format(train.keyword.isin(['debris', 'wreckage', 'derailment']).sum()))
sbn.barplot(y='keyword', x='frequencies', data=groupby_keyword.iloc[-20:], color='gray')
plt.gca().set_xlabel('Frequency of disasters')
plt.gca().set_ylabel('Keywords')
plt.suptitle('20 lowest frequencies of disaster by keyword', size=15)

print('The 20 keywords associated with the lowest frequencies of disaster occur between {} and {} times with an average of {}.'.format(groupby_keyword[-20:]['count'].min(), groupby_keyword[-20:]['count'].max(), groupby_keyword[-20:]['count'].mean()))
groupby_location = train[['location', 'target']].groupby('location')['target'].agg(frequencies='mean', count='size').reset_index().sort_values(by='frequencies', ascending=False)
x = groupby_location.frequencies.apply(lambda x: np.round(x, 1)).value_counts().index
y = groupby_location.frequencies.apply(lambda x: np.round(x, 1)).value_counts()
y = y / np.sum(y)
y *= 100
sbn.barplot(x=x, y=y, color='gray')
plt.gca().set_xlabel('Frequency of disasters')
plt.gca().set_ylabel('Proportion of locations')
plt.gca().set_yticklabels([])
plt.gca().set_xbound(-0.5, 11)
plt.gca().set_ybound(0, 65)
for i in range(11):
    plt.annotate(str(round(y.loc[i / 10], 1)) + '%', xy=(i - 0.4, y.loc[i / 10] + 1), size=10)
plt.suptitle('Proportion of locations by frequency of disasters', size=15)

x = groupby_location['count'].value_counts()[:10].index
y = groupby_location['count'].value_counts().to_numpy()
y = y[:10] / np.sum(y)
y *= 100
sbn.barplot(x=x, y=y, color='gray')
plt.gca().set_ylim([0, 100])
plt.gca().set_xlabel('Number of occurrences')
plt.gca().set_ylabel('Proportion of locations')
for i in range(10):
    plt.annotate(str(round(y[i], 1)) + '%', xy=(i - 0.3, y[i] + 2), size=10)
plt.suptitle('Proportion of locations by number of occurrences in the training set', size=15)

print('Besides, {}% of the locations appear more than 11 times in the training set.'.format(round(100 - np.sum(y), 1)))
groupby_location[groupby_location['count'] >= 20]
keyword_location = train.copy()
keyword_location['keyword_location'] = keyword_location.keyword + '_' + keyword_location.location
pd.DataFrame(keyword_location.keyword_location.value_counts().reset_index().to_numpy(), columns=['(keyword, location) pairs', 'occurrences'])
keyword_location[keyword_location.keyword_location == 'sandstorm_USA']

def ngram_occurrences(corpus=train.text, i=0, j=20, stop_words=None, ngram_range=(1, 1), tokenizer=None):
    """ Function to return a dataframe containing some chosen n-grams and, for each n-grams, the number of times it appear in the corpus. 
      The defaults values return a the barplot for the 20 most frequent n-grams.

        Parameters
        ----------
        corpus : Series (default=train.text)
            A Series containing the tweets.
        i : Integer (default=0)
            The minimum index of the n-grams to draw. The n-grams are sorted by number of occurrences (with the index starting at 0). So 0 stands for the most frequent n-gram, 
            1 for the second most frequent n-gram, etc.
        j : Integer (default=20)
            1 + the maximum index of the n-grams to draw. The n-grams are sorted by number of occurrences (with the index starting at 0). So j=20 stands for the 19th index, wich stands for the
             20th most frequent n-gram.
        stop_words : Iterable (default=None)
            If not None, the stop words to remove from the tokens.
        ngram_range : Tuple (min_n, max_n) (default=(1, 1))
            The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used.
            For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams. Only applies if analyzer is not callable.
        tokenizer : Tokenizer (default=None)
            If not None, a tokenizer to use instead of the default tokenizer.

        """