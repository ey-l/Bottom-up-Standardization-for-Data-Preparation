import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1['target'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(_input1['target'])
_input1['target'].value_counts().head(10).plot.pie(autopct='%1.1f%%')
_input1['location'].value_counts()[:10].plot(kind='bar')
_input1['keyword'].value_counts()[:10].plot(kind='bar')
_input1[_input1['target'] == 1]['location'].value_counts()[:10].plot(kind='bar')

def wl(text):
    return len(text.split(' '))
_input1['word_length'] = _input1['text'].apply(wl)
_input1['word_length'].hist()
sns.kdeplot(_input1[_input1['target'] == 1]['word_length'], color='g')
sns.kdeplot(_input1[_input1['target'] == 0]['word_length'], color='r')
plt.legend(['disaster', 'real'])
sns.barplot(x='target', y='word_length', data=_input1)
_input1['char_length'] = _input1['text'].apply(len)
_input1['char_length'].hist()
sns.kdeplot(_input1[_input1['target'] == 1]['char_length'], color='g')
sns.kdeplot(_input1[_input1['target'] == 0]['char_length'], color='r')
plt.legend(['disaster', 'real'])
sns.barplot(x='target', y='char_length', data=_input1)
sns.scatterplot(x='char_length', y='word_length', data=_input1)
from scipy import stats
import statsmodels.api as sm
stats.probplot(_input1['char_length'], plot=plt)
stats.probplot(_input1['word_length'], plot=plt)
_input1['unique_word_count'] = _input1['text'].apply(lambda x: len(set(str(x).split())))
_input1['unique_word_count'].hist()
sns.kdeplot(_input1[_input1['target'] == 1]['unique_word_count'], color='g')
sns.kdeplot(_input1[_input1['target'] == 0]['unique_word_count'], color='r')
plt.legend(['disaster', 'real'])
sns.scatterplot(x='unique_word_count', y='word_length', data=_input1)
sns.scatterplot(x='char_length', y='unique_word_count', data=_input1)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
all_stopwords = stopwords.words('english')
_input1['stop_words'] = _input1['text'].apply(lambda x: len([words for words in str(x).lower().split() if words in all_stopwords]))
_input1['stop_words'].hist()
sns.barplot(x='target', y='stop_words', data=_input1)
sns.kdeplot(_input1[_input1['target'] == 1]['stop_words'], color='g')
sns.kdeplot(_input1[_input1['target'] == 0]['stop_words'], color='r')
plt.legend(['disaster', 'real'])
features = ['word_length', 'char_length', 'unique_word_count']
for i in features:
    sns.scatterplot(x=i, y='stop_words', data=_input1)
corr = _input1.corr()
sns.heatmap(corr, annot=True)
_input1['text']
from nltk.util import ngrams

def get_bigram(text):
    big = ''
    token = nltk.word_tokenize(text)
    big = list(ngrams(token, 2))
    return str(big)
_input1['bigram'] = _input1['text'].apply(get_bigram)
_input1['bigram']