import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1['target'].value_counts()
_input1['length'] = np.NaN
for i in range(0, len(_input1['text'])):
    _input1['length'][i] = len(_input1['text'][i])
_input1.length = _input1.length.astype(int)
_input1.head()
sns.set_style('darkgrid')
(f, (ax1, ax2)) = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, tight_layout=True)
sns.distplot(_input1[_input1['target'] == 1]['length'], bins=30, ax=ax1)
sns.distplot(_input1[_input1['target'] == 0]['length'], bins=30, ax=ax2)
ax1.set_title('\n Distribution of length of tweet labelled Disaster \n')
ax2.set_title('\n Distribution of length of tweet labelled No Disaster \n')
ax1.set_ylabel('Frequency')
text = ' '.join((post for post in _input1[_input1['target'] == 1].text))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Frequntly occuring words related to Disaster \n\n', fontsize=18)
plt.axis('off')
text = ' '.join((post for post in _input1[_input1['target'] == 0].text))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Frequntly occuring words related to No Disaster \n\n', fontsize=18)
plt.axis('off')
_input1['target'].value_counts(normalize=True)
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\\w+')
_input1.loc[:, 'text'] = _input1.text.apply(lambda x: str.lower(x))
_input1['text'] = _input1['text'].str.replace('http.*.*', '', regex=True)
_input1['text'] = _input1['text'].str.replace('û.*.*', '', regex=True)
_input1['text'] = _input1['text'].str.replace('\\d+', '', regex=True)
_input1['tokens'] = _input1['text'].map(tokenizer.tokenize)
_input1.head()
print(stopwords.words('english'))
stop = stopwords.words('english')
item = ['amp']
stop.extend(item)
_input1['tokens'] = _input1['tokens'].apply(lambda x: [item for item in x if item not in stop])
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
lemmatize_words = []
for i in range(len(_input1['tokens'])):
    word = ''
    for j in range(len(_input1['tokens'][i])):
        lemm_word = lemmatizer.lemmatize(_input1['tokens'][i][j])
        word = word + ' ' + lemm_word
    lemmatize_words.append(word)
_input1['lemmatized'] = lemmatize_words
_input1.head()
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0.head()
_input0['length'] = np.NaN
for i in range(0, len(_input0['text'])):
    _input0['length'][i] = len(_input0['text'][i])
_input0.length = _input0.length.astype(int)
text = ' '.join((post for post in _input1.text))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('\nFrequntly occuring words in test dataframe \n\n', fontsize=18)
plt.axis('off')
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\\w+')
_input0.loc[:, 'text'] = _input0.text.apply(lambda x: str.lower(x))
_input0['text'] = _input0['text'].str.replace('http.*.*', '', regex=True)
_input0['text'] = _input0['text'].str.replace('û.*.*', '', regex=True)
_input0['text'] = _input0['text'].str.replace('\\d+', '', regex=True)
_input0['tokens'] = _input0['text'].map(tokenizer.tokenize)
_input0.head()
_input0['tokens'] = _input0['tokens'].apply(lambda x: [item for item in x if item not in stop])
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
lemmatize_words = []
for i in range(len(_input0['tokens'])):
    word = ''
    for j in range(len(_input0['tokens'][i])):
        lemm_word = lemmatizer.lemmatize(_input0['tokens'][i][j])
        word = word + ' ' + lemm_word
    lemmatize_words.append(word)
_input0['lemmatized'] = lemmatize_words
_input0.head()
text = ' '.join((post for post in _input0.lemmatized))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('\n Frequntly occuring words in test dataframe after lemmatizing \n\n', fontsize=18)
plt.axis('off')
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(_input1['lemmatized'])
test_vectors = vectorizer.transform(_input0['lemmatized'])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
X = _input1['lemmatized']
y = _input1['target']
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X)
X_test = vectorizer.transform(_input0['lemmatized'])
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)