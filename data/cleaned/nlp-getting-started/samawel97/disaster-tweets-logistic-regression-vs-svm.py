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
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.head()
df['target'].value_counts()
df['length'] = np.NaN
for i in range(0, len(df['text'])):
    df['length'][i] = len(df['text'][i])
df.length = df.length.astype(int)
df.head()
sns.set_style('darkgrid')
(f, (ax1, ax2)) = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, tight_layout=True)
sns.distplot(df[df['target'] == 1]['length'], bins=30, ax=ax1)
sns.distplot(df[df['target'] == 0]['length'], bins=30, ax=ax2)
ax1.set_title('\n Distribution of length of tweet labelled Disaster \n')
ax2.set_title('\n Distribution of length of tweet labelled No Disaster \n')
ax1.set_ylabel('Frequency')
text = ' '.join((post for post in df[df['target'] == 1].text))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Frequntly occuring words related to Disaster \n\n', fontsize=18)
plt.axis('off')

text = ' '.join((post for post in df[df['target'] == 0].text))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Frequntly occuring words related to No Disaster \n\n', fontsize=18)
plt.axis('off')

df['target'].value_counts(normalize=True)
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\\w+')
df.loc[:, 'text'] = df.text.apply(lambda x: str.lower(x))
df['text'] = df['text'].str.replace('http.*.*', '', regex=True)
df['text'] = df['text'].str.replace('û.*.*', '', regex=True)
df['text'] = df['text'].str.replace('\\d+', '', regex=True)
df['tokens'] = df['text'].map(tokenizer.tokenize)
df.head()
print(stopwords.words('english'))
stop = stopwords.words('english')
item = ['amp']
stop.extend(item)
df['tokens'] = df['tokens'].apply(lambda x: [item for item in x if item not in stop])
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
lemmatize_words = []
for i in range(len(df['tokens'])):
    word = ''
    for j in range(len(df['tokens'][i])):
        lemm_word = lemmatizer.lemmatize(df['tokens'][i][j])
        word = word + ' ' + lemm_word
    lemmatize_words.append(word)
df['lemmatized'] = lemmatize_words
df.head()
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
test.head()
test['length'] = np.NaN
for i in range(0, len(test['text'])):
    test['length'][i] = len(test['text'][i])
test.length = test.length.astype(int)
text = ' '.join((post for post in df.text))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('\nFrequntly occuring words in test dataframe \n\n', fontsize=18)
plt.axis('off')

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\\w+')
test.loc[:, 'text'] = test.text.apply(lambda x: str.lower(x))
test['text'] = test['text'].str.replace('http.*.*', '', regex=True)
test['text'] = test['text'].str.replace('û.*.*', '', regex=True)
test['text'] = test['text'].str.replace('\\d+', '', regex=True)
test['tokens'] = test['text'].map(tokenizer.tokenize)
test.head()
test['tokens'] = test['tokens'].apply(lambda x: [item for item in x if item not in stop])
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
lemmatize_words = []
for i in range(len(test['tokens'])):
    word = ''
    for j in range(len(test['tokens'][i])):
        lemm_word = lemmatizer.lemmatize(test['tokens'][i][j])
        word = word + ' ' + lemm_word
    lemmatize_words.append(word)
test['lemmatized'] = lemmatize_words
test.head()
text = ' '.join((post for post in test.lemmatized))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('\n Frequntly occuring words in test dataframe after lemmatizing \n\n', fontsize=18)
plt.axis('off')

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(df['lemmatized'])
test_vectors = vectorizer.transform(test['lemmatized'])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
X = df['lemmatized']
y = df['target']
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X)
X_test = vectorizer.transform(test['lemmatized'])
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)