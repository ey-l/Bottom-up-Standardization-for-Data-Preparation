import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from matplotlib import rcParams
from wordcloud import WordCloud
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
sns.set_style('darkgrid')
(f, (ax1, ax2)) = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, tight_layout=True)
sns.distplot(_input1[_input1['target'] == 1]['length'], bins=30, ax=ax1)
sns.distplot(_input1[_input1['target'] == 0]['length'], bins=30, ax=ax2)
ax1.set_title('\n Distribution of length of tweet labelled Disaster\n')
ax2.set_title('\nDistribution of length of tweet labelled No Disaster\n ')
ax1.set_ylabel('Frequency')
text = ' '.join((post for post in _input1[_input1['target'] == 1].text))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('\nFrequntly occuring words related to Disaster \n\n', fontsize=18)
plt.axis('off')
text = ' '.join((post for post in _input1[_input1['target'] == 0].text))
wordcloud = WordCloud(max_font_size=90, max_words=50, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('\nFrequntly occuring words related to No Disaster \n\n', fontsize=18)
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
X = _input1['lemmatized']
y = _input1['target']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, random_state=42)
y_train.value_counts()
y_test.shape
pipe = Pipeline([('cvec', CountVectorizer()), ('lr', LogisticRegression())])
tuned_params = {'cvec__max_features': [2500, 3000, 3500], 'cvec__min_df': [2, 3], 'cvec__max_df': [0.9, 0.95], 'cvec__ngram_range': [(1, 1), (1, 2)]}
gs = GridSearchCV(pipe, param_grid=tuned_params, cv=3)