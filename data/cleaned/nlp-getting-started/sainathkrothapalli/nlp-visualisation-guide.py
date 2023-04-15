import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/nlp-getting-started/train.csv')
data.head()
data['target'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data['target'])
data['target'].value_counts().head(10).plot.pie(autopct='%1.1f%%')
data['location'].value_counts()[:10].plot(kind='bar')
data['keyword'].value_counts()[:10].plot(kind='bar')
data[data['target'] == 1]['location'].value_counts()[:10].plot(kind='bar')

def wl(text):
    return len(text.split(' '))
data['word_length'] = data['text'].apply(wl)
data['word_length'].hist()
sns.kdeplot(data[data['target'] == 1]['word_length'], color='g')
sns.kdeplot(data[data['target'] == 0]['word_length'], color='r')
plt.legend(['disaster', 'real'])
sns.barplot(x='target', y='word_length', data=data)
data['char_length'] = data['text'].apply(len)
data['char_length'].hist()
sns.kdeplot(data[data['target'] == 1]['char_length'], color='g')
sns.kdeplot(data[data['target'] == 0]['char_length'], color='r')
plt.legend(['disaster', 'real'])
sns.barplot(x='target', y='char_length', data=data)
sns.scatterplot(x='char_length', y='word_length', data=data)
from scipy import stats
import statsmodels.api as sm
stats.probplot(data['char_length'], plot=plt)
stats.probplot(data['word_length'], plot=plt)
data['unique_word_count'] = data['text'].apply(lambda x: len(set(str(x).split())))
data['unique_word_count'].hist()
sns.kdeplot(data[data['target'] == 1]['unique_word_count'], color='g')
sns.kdeplot(data[data['target'] == 0]['unique_word_count'], color='r')
plt.legend(['disaster', 'real'])
sns.scatterplot(x='unique_word_count', y='word_length', data=data)
sns.scatterplot(x='char_length', y='unique_word_count', data=data)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
all_stopwords = stopwords.words('english')
data['stop_words'] = data['text'].apply(lambda x: len([words for words in str(x).lower().split() if words in all_stopwords]))
data['stop_words'].hist()
sns.barplot(x='target', y='stop_words', data=data)
sns.kdeplot(data[data['target'] == 1]['stop_words'], color='g')
sns.kdeplot(data[data['target'] == 0]['stop_words'], color='r')
plt.legend(['disaster', 'real'])
features = ['word_length', 'char_length', 'unique_word_count']
for i in features:
    sns.scatterplot(x=i, y='stop_words', data=data)

corr = data.corr()
sns.heatmap(corr, annot=True)
data['text']
from nltk.util import ngrams

def get_bigram(text):
    big = ''
    token = nltk.word_tokenize(text)
    big = list(ngrams(token, 2))
    return str(big)
data['bigram'] = data['text'].apply(get_bigram)
data['bigram']