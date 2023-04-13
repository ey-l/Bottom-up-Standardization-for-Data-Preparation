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
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
print('Shape of train data : ', _input1.shape)
print('Shape of test data : ', _input0.shape)
_input1.head()
print('Missing value in train data :\n', _input1.isna().sum())
print('\nMissing value in test data :\n', _input0.isna().sum())
_input1[['id', 'target']].groupby('target').count()
counts = pd.DataFrame(_input1['target'].value_counts())
counts = counts.rename(columns={'target': 'Samples'}, index={0: 'Not Real', 1: 'Real'}, inplace=False)
ax = sns.barplot(x=counts.index, y=counts.Samples)
for p in ax.patches:
    height = p.get_height()
    ax.text(x=p.get_x() + p.get_width() / 2, y=height, s=round(height), ha='center')
_input1['tweet_len'] = _input1.apply(lambda row: len(row['text']), axis=1)
_input0['tweet_len'] = _input0.apply(lambda row: len(row['text']), axis=1)
_input0.head()
plt.figure(figsize=(10, 6))
_input1[_input1.target == 0].tweet_len.plot(bins=40, kind='hist', color='blue', label='irrelevant', alpha=0.6)
_input1[_input1.target == 1].tweet_len.plot(bins=40, kind='hist', color='red', label='relevant', alpha=0.6)
plt.legend()
plt.xlabel('Length of text')

def text_cleaning_process(text):
    STOPWORDS = stopwords.words('english')
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
_input1['clean_text'] = _input1.apply(lambda row: text_cleaning_process(row['text']), axis=1)
_input0['clean_text'] = _input0.apply(lambda row: text_cleaning_process(row['text']), axis=1)
_input1.head()
X_train = _input1['clean_text'].values
y_train = _input1['target'].values
X_test = _input0['clean_text'].values
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
(xtrain, xvalid, ytrain, yvalid) = train_test_split(X_train, y_train, stratify=y_train, random_state=42, test_size=0.3, shuffle=True)
print(xtrain.shape)
print(xvalid.shape)
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern='\\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')