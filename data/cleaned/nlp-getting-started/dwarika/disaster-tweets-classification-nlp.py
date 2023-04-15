import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
sample = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
print('Shape of train data : ', train.shape)
print('Shape of test data : ', test.shape)
train.head()
print('Missing value in train data :\n', train.isna().sum())
print('\nMissing value in test data :\n', test.isna().sum())
train[['id', 'target']].groupby('target').count()
counts = pd.DataFrame(train['target'].value_counts())
counts.rename(columns={'target': 'Samples'}, index={0: 'Not Real', 1: 'Real'}, inplace=True)
ax = sns.barplot(x=counts.index, y=counts.Samples)
for p in ax.patches:
    height = p.get_height()
    ax.text(x=p.get_x() + p.get_width() / 2, y=height, s=round(height), ha='center')
train['tweet_len'] = train.apply(lambda row: len(row['text']), axis=1)
test['tweet_len'] = test.apply(lambda row: len(row['text']), axis=1)
test.head()
plt.figure(figsize=(10, 6))
train[train.target == 0].tweet_len.plot(bins=40, kind='hist', color='blue', label='irrelevant', alpha=0.6)
train[train.target == 1].tweet_len.plot(bins=40, kind='hist', color='red', label='relevant', alpha=0.6)
plt.legend()
plt.xlabel('Length of text')


def text_cleaning_process(text):
    STOPWORDS = stopwords.words('english')
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
train['clean_text'] = train.apply(lambda row: text_cleaning_process(row['text']), axis=1)
test['clean_text'] = test.apply(lambda row: text_cleaning_process(row['text']), axis=1)
train.head()
X_train = train['clean_text'].values
y_train = train['target'].values
X_test = test['clean_text'].values
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
(xtrain, xvalid, ytrain, yvalid) = train_test_split(X_train, y_train, stratify=y_train, random_state=42, test_size=0.3, shuffle=True)
print(xtrain.shape)
print(xvalid.shape)
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern='\\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')