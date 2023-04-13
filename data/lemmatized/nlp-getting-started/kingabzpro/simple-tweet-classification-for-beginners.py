import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
print('Training Data')
print('Testing Data')
print('Train Dataset shape:\n', _input1.shape, '\n')
print('Test Dataset shape:\n', _input0.shape)
print('Train Dataset missing data:\n', _input1.isnull().sum(), '\n')
print('Test Dataset missing data:\n', _input0.isnull().sum())
VCtrain = _input1['target'].value_counts().to_frame()
sns.barplot(data=VCtrain, x=VCtrain.index, y='target', palette='viridis')
VCtrain
common_keywords = _input1['keyword'].value_counts()[:20].to_frame()
fig = plt.figure(figsize=(15, 6))
sns.barplot(data=common_keywords, x=common_keywords.index, y='keyword', palette='viridis')
plt.title('Most common keywords', size=16)
plt.xticks(rotation=70, size=12)
_input1[_input1.text.str.contains('disaster')].target.value_counts().to_frame().rename(index={1: 'Disaster', 0: 'normal'}).plot.pie(y='target', figsize=(12, 6), title='Tweets with Disaster mentioned')
_input1.location.value_counts()[:10].to_frame()
_input1.text = _input1.text.apply(lambda x: x.lower())
_input0.text = _input0.text.apply(lambda x: x.lower())
_input1.text = _input1.text.apply(lambda x: re.sub('\\[.*?\\]', '', x))
_input0.text = _input0.text.apply(lambda x: re.sub('\\[.*?\\]', '', x))
_input1.text = _input1.text.apply(lambda x: re.sub('<.*?>+', '', x))
_input0.text = _input0.text.apply(lambda x: re.sub('<.*?>+', '', x))
_input1.text = _input1.text.apply(lambda x: re.sub('https?://\\S+|www\\.\\S+', '', x))
_input0.text = _input0.text.apply(lambda x: re.sub('https?://\\S+|www\\.\\S+', '', x))
_input1.text = _input1.text.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
_input0.text = _input0.text.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
_input1.text = _input1.text.apply(lambda x: re.sub('\n', '', x))
_input0.text = _input0.text.apply(lambda x: re.sub('\n', '', x))
_input1.text = _input1.text.apply(lambda x: re.sub('\\w*\\d\\w*', '', x))
_input0.text = _input0.text.apply(lambda x: re.sub('\\w*\\d\\w*', '', x))
_input1.text.head()
disaster_tweets = _input1[_input1['target'] == 1]['text']
non_disaster_tweets = _input1[_input1['target'] == 0]['text']
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=[16, 8])
wordcloud1 = WordCloud(background_color='white', width=600, height=400).generate(' '.join(disaster_tweets))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Disaster Tweets', fontsize=40)
wordcloud2 = WordCloud(background_color='white', width=600, height=400).generate(' '.join(non_disaster_tweets))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Non Disaster Tweets', fontsize=40)
token = nltk.tokenize.RegexpTokenizer('\\w+')
_input1.text = _input1.text.apply(lambda x: token.tokenize(x))
_input0.text = _input0.text.apply(lambda x: token.tokenize(x))
nltk.download('stopwords')
_input1.text = _input1.text.apply(lambda x: [w for w in x if w not in stopwords.words('english')])
_input0.text = _input0.text.apply(lambda x: [w for w in x if w not in stopwords.words('english')])
_input1.text.head()
_input0.text.head()
stemmer = nltk.stem.PorterStemmer()
_input1.text = _input1.text.apply(lambda x: ' '.join((stemmer.stem(token) for token in x)))
_input0.text = _input0.text.apply(lambda x: ' '.join((stemmer.stem(token) for token in x)))
_input1.text.head()
count_vectorizer = CountVectorizer()
train_vectors_count = count_vectorizer.fit_transform(_input1['text'])
test_vectors_count = count_vectorizer.transform(_input0['text'])
CLR = LogisticRegression(C=2)
scores = cross_val_score(CLR, train_vectors_count, _input1['target'], cv=6, scoring='f1')
scores
NB_Vec = MultinomialNB()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
scores = cross_val_score(NB_Vec, train_vectors_count, _input1['target'], cv=cv, scoring='f1')
scores