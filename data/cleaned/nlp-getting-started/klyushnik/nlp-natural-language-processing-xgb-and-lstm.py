import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
test.head()
train.head()
train.info()
test.info()
print('Missing values train', train.isnull().sum(), sep='')
print('Missing values test', train.isnull().sum(), sep='')
train.fillna(' ', inplace=True)
test.fillna(' ', inplace=True)
test.head()
from wordcloud import WordCloud, STOPWORDS

def str_corpus(corpus):
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus

def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus

def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='white', stopwords=STOPWORDS, width=3000, height=2500, max_words=200, random_state=42).generate(str_corpus(corpus))
    return wordCloud
corpus = get_corpus(train['text'].values)
procWordCloud = get_wordCloud(corpus)
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.imshow(procWordCloud)
plt.axis('off')
plt.subplot(1, 2, 1)

def str_corpus(corpus):
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus

def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus

def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='white', stopwords=STOPWORDS, width=3000, height=2500, max_words=200, random_state=42).generate(str_corpus(corpus))
    return wordCloud
corpus = get_corpus(test['text'].values)
procWordCloud = get_wordCloud(corpus)
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.imshow(procWordCloud)
plt.axis('off')
plt.subplot(1, 2, 1)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation
english_stopwords = stopwords.words('english')

def remove_punct(text):
    table = {33: ' ', 34: ' ', 35: ' ', 36: ' ', 37: ' ', 38: ' ', 39: ' ', 40: ' ', 41: ' ', 42: ' ', 43: ' ', 44: ' ', 45: ' ', 46: ' ', 47: ' ', 58: ' ', 59: ' ', 60: ' ', 61: ' ', 62: ' ', 63: ' ', 64: ' ', 91: ' ', 92: ' ', 93: ' ', 94: ' ', 95: ' ', 96: ' ', 123: ' ', 124: ' ', 125: ' ', 126: ' '}
    return text.translate(table)
train['text'] = train['text'].map(lambda x: x.lower())
train['text'] = train['text'].map(lambda x: remove_punct(x))
train['text'] = train['text'].map(lambda x: x.split(' '))
train['text'] = train['text'].map(lambda x: [token for token in x if token not in english_stopwords and token != ' ' and (token.strip() not in punctuation)])
train['text'] = train['text'].map(lambda x: ' '.join(x))
train.head()
test['text'] = test['text'].map(lambda x: x.lower())
test['text'] = test['text'].map(lambda x: remove_punct(x))
test['text'] = test['text'].map(lambda x: x.split(' '))
test['text'] = test['text'].map(lambda x: [token for token in x if token not in english_stopwords and token != ' ' and (token.strip() not in punctuation)])
test['text'] = test['text'].map(lambda x: ' '.join(x))
test.head()
train['text_total'] = train['keyword'] + train['location'] + train['text']
test['text_total'] = test['keyword'] + test['location'] + test['text']
train.head()
df_train = train.copy(deep=True)
df_test = test.copy(deep=True)
df_train = df_train.drop(df_train[['id', 'keyword', 'location', 'text']], axis=1)
df_test = df_test.drop(df_test[['id', 'keyword', 'location', 'text']], axis=1)
df_train.head()

def str_corpus(corpus):
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus

def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus

def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='white', stopwords=STOPWORDS, width=3000, height=2500, max_words=200, random_state=42).generate(str_corpus(corpus))
    return wordCloud
corpus = get_corpus(train['text'].values)
procWordCloud = get_wordCloud(corpus)
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.imshow(procWordCloud)
plt.axis('off')
plt.subplot(1, 2, 1)
(X_train, X_test, y_train, y_test) = train_test_split(train['text_total'], train['target'], test_size=0.4, random_state=42)
sgd_ppl_clf = Pipeline([('tfidf', TfidfVectorizer()), ('sgd_clf', SGDClassifier(random_state=42))])
knb_ppl_clf = Pipeline([('tfidf', TfidfVectorizer()), ('knb_clf', KNeighborsClassifier(n_neighbors=10))])