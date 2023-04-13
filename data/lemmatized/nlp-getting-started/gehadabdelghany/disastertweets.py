import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.stem.snowball import SnowballStemmer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1.head()
_input0.head()
(_input1.shape, _input0.shape)
_input1.duplicated().sum()
_input0.duplicated().sum()
_input1.columns
_input0.columns
_input1['location'].value_counts()
_input0['location'].value_counts()
_input1['keyword'].value_counts()
_input0['keyword'].value_counts()
_input0 = _input0.drop(['location'], axis=1, inplace=False)
_input1.isnull().sum()
_input0.isnull().sum()
print('Disaster_Tweets_numbers: ' + str(len(_input1[_input1['target'] == 1])))
print('not Disaster_Tweets_numbers: ' + str(len(_input1[_input1['target'] == 0])))
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('[^\\d\\W]+')
_input1['cleaned'] = [tokenizer.tokenize(item) for item in _input1['text']]
_input0['cleaned'] = [tokenizer.tokenize(item) for item in _input0['text']]
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
_input1['cleaned'] = _input1['cleaned'].apply(lambda x: [stemmer.stem(y) for y in x])
_input0['cleaned'] = _input0['cleaned'].apply(lambda x: [stemmer.stem(y) for y in x])
_input1['cleaned']
from nltk.corpus import stopwords
stop = stopwords.words('english')
_input1['cleaned'] = [item for item in _input1['cleaned'] if item not in stop]
_input0['cleaned'] = [item for item in _input0['cleaned'] if item not in stop]
_input0['cleaned']
_input1['cleaned']

def notT(text):
    text = text.apply(lambda x: [item for item in x if len(item) > 3])
    return text
_input1['cleaned'] = notT(_input1['cleaned'])
_input0['cleaned'] = notT(_input0['cleaned'])
_input0['cleaned']
_input1['cleaned'] = _input1['cleaned'].apply(', '.join)
_input0['cleaned'] = _input0['cleaned'].apply(', '.join)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from random import choice
stopwords = set(STOPWORDS)

def wordCloud(tokens, plot=1):
    lemtz = WordNetLemmatizer()
    lemmatize_keywords = []
    for token in tokens:
        lemmatize_keywords.append(lemtz.lemmatize(token, wordnet.VERB))
    if plot == 1:
        plot_wordcloud(lemmatize_keywords)
    else:
        return ' '.join(lemmatize_keywords)

def plot_wordcloud(text, bg_color='salmon', cmap='rainbow'):
    c = choice(['Paired', 'Set2', 'husl', 'Spectral', 'coolwarm'])
    (fig, (ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    wordcloud = WordCloud(width=3000, height=2000, background_color=bg_color, colormap=cmap, collocations=False, stopwords=STOPWORDS, random_state=51).generate(' '.join(text))
    ax1.imshow(wordcloud)
    ax1.axis('off')
    labels = pd.Series(data=text).value_counts().index[:20]
    data = pd.Series(data=text).value_counts()[:20]
    sns.barplot(y=labels, x=data, ax=ax2, palette=c)
key_data = _input1['keyword'].fillna('blank').apply(lambda x: re.sub('[^a-zA-Z]+', '_', x))
keywords_list = []
for keyword in key_data:
    if keyword != 'blank':
        keywords_list.extend(keyword.split())
wordCloud(keywords_list)
key_data = _input1[_input1['target'] == 0]['keyword'].fillna('blank').apply(lambda x: re.sub('[^a-zA-Z]+', '_', x))
keywords_list = []
for keyword in key_data:
    if keyword != 'blank':
        keywords_list.extend(keyword.split())
plot_wordcloud(keywords_list)
key_data = _input1[_input1['target'] == 1]['keyword'].fillna('blank').apply(lambda x: re.sub('[^a-zA-Z]+', '_', x))
keywords_list = []
for keyword in key_data:
    if keyword != 'blank':
        keywords_list.extend(keyword.split())
plot_wordcloud(keywords_list)
plt.figure(figsize=(20, 6))
labels = _input1['location'].value_counts().index[:20]
data = _input1['location'].value_counts()[:20]
ax = sns.barplot(x=labels, y=data)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + 0.2, p.get_height() + 0.5))
ax.set_title('Top 20 Country who tweets', fontsize=20)
ax.set_xticklabels(labels=labels, rotation=45)
plt.figure(figsize=(20, 6))
labels = _input1[_input1['target'] == 1]['location'].value_counts().index[:20]
data = _input1[_input1['target'] == 1]['location'].value_counts()[:20]
ax = sns.barplot(x=labels, y=data)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + 0.2, p.get_height() + 0.5))
ax.set_title('Top 20 Country who tweets Disaster', fontsize=20)
ax.set_xticklabels(labels=labels, rotation=45)
plt.figure(figsize=(20, 6))
data = _input1[_input1['target'] == 0]['location'].value_counts()[:20]
labels = _input1[_input1['target'] == 0]['location'].value_counts().index[:20]
ax = sns.barplot(x=labels, y=data)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + 0.2, p.get_height() + 0.5))
ax.set_title('Top 20 Country who tweets Non Disaster', fontsize=20)
ax.set_xticklabels(labels=labels, rotation=90)
from sklearn.model_selection import train_test_split
(X, y) = (_input1['text'], _input1['target'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()
len(vectorizer.get_feature_names())

def evaluate(y_true, y_predicted):
    acc = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    return (acc, precision, recall, f1)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix