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
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
print('Training Data')

print('Testing Data')

print('Train Dataset shape:\n', train.shape, '\n')
print('Test Dataset shape:\n', test.shape)
print('Train Dataset missing data:\n', train.isnull().sum(), '\n')
print('Test Dataset missing data:\n', test.isnull().sum())
VCtrain = train['target'].value_counts().to_frame()
sns.barplot(data=VCtrain, x=VCtrain.index, y='target', palette='viridis')
VCtrain


common_keywords = train['keyword'].value_counts()[:20].to_frame()
fig = plt.figure(figsize=(15, 6))
sns.barplot(data=common_keywords, x=common_keywords.index, y='keyword', palette='viridis')
plt.title('Most common keywords', size=16)
plt.xticks(rotation=70, size=12)
train[train.text.str.contains('disaster')].target.value_counts().to_frame().rename(index={1: 'Disaster', 0: 'normal'}).plot.pie(y='target', figsize=(12, 6), title='Tweets with Disaster mentioned')
train.location.value_counts()[:10].to_frame()
train.text = train.text.apply(lambda x: x.lower())
test.text = test.text.apply(lambda x: x.lower())
train.text = train.text.apply(lambda x: re.sub('\\[.*?\\]', '', x))
test.text = test.text.apply(lambda x: re.sub('\\[.*?\\]', '', x))
train.text = train.text.apply(lambda x: re.sub('<.*?>+', '', x))
test.text = test.text.apply(lambda x: re.sub('<.*?>+', '', x))
train.text = train.text.apply(lambda x: re.sub('https?://\\S+|www\\.\\S+', '', x))
test.text = test.text.apply(lambda x: re.sub('https?://\\S+|www\\.\\S+', '', x))
train.text = train.text.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
test.text = test.text.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
train.text = train.text.apply(lambda x: re.sub('\n', '', x))
test.text = test.text.apply(lambda x: re.sub('\n', '', x))
train.text = train.text.apply(lambda x: re.sub('\\w*\\d\\w*', '', x))
test.text = test.text.apply(lambda x: re.sub('\\w*\\d\\w*', '', x))
train.text.head()
disaster_tweets = train[train['target'] == 1]['text']
non_disaster_tweets = train[train['target'] == 0]['text']
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
train.text = train.text.apply(lambda x: token.tokenize(x))
test.text = test.text.apply(lambda x: token.tokenize(x))

nltk.download('stopwords')
train.text = train.text.apply(lambda x: [w for w in x if w not in stopwords.words('english')])
test.text = test.text.apply(lambda x: [w for w in x if w not in stopwords.words('english')])
train.text.head()
test.text.head()
stemmer = nltk.stem.PorterStemmer()
train.text = train.text.apply(lambda x: ' '.join((stemmer.stem(token) for token in x)))
test.text = test.text.apply(lambda x: ' '.join((stemmer.stem(token) for token in x)))
train.text.head()
count_vectorizer = CountVectorizer()
train_vectors_count = count_vectorizer.fit_transform(train['text'])
test_vectors_count = count_vectorizer.transform(test['text'])
CLR = LogisticRegression(C=2)
scores = cross_val_score(CLR, train_vectors_count, train['target'], cv=6, scoring='f1')
scores
NB_Vec = MultinomialNB()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
scores = cross_val_score(NB_Vec, train_vectors_count, train['target'], cv=cv, scoring='f1')
scores