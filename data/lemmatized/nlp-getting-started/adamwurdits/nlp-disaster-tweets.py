import numpy as np
import pandas as pd
import re
from textblob import TextBlob
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, model_selection
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head(20)
sns.countplot(_input1['target'])
_input1['target'].value_counts() / len(_input1)
tweets = []
for tweet in _input1['text']:
    tweets += [tweet]
num_words = []
num_stop_words = []
polarity = []
subjectivity = []
for tweet in tweets:
    num_words += [len(tweet.split())]
    num_stop_words += [len([stopword for stopword in tweet.lower().split() if stopword in STOPWORDS])]
    tweet_blob = TextBlob(tweet)
    polarity += [tweet_blob.sentiment.polarity]
    subjectivity += [tweet_blob.sentiment.subjectivity]
_input1['length'] = _input1['text'].str.len()
_input1['num_words'] = num_words
_input1['num_stop_words'] = num_stop_words
_input1['polarity'] = polarity
_input1['subjectivity'] = subjectivity
_input1.groupby('target')['length', 'num_words', 'num_stop_words', 'polarity', 'subjectivity'].mean()
regular_tweets = _input1[_input1['target'] == 0]['text'].to_list()
disaster_tweets = _input1[_input1['target'] == 1]['text'].to_list()
joined_regular_tweets = ' '.join(regular_tweets)
joined_disaster_tweets = ' '.join(disaster_tweets)
regular_cloud = WordCloud().generate(joined_regular_tweets)
disaster_cloud = WordCloud().generate(joined_disaster_tweets)
fig = plt.figure(figsize=(16, 12))
fig.add_subplot(221)
plt.title('Regular tweets')
plt.imshow(regular_cloud)
fig.add_subplot(222)
plt.title('Disaster tweets')
plt.imshow(disaster_cloud)
vect = CountVectorizer(max_features=100, ngram_range=(1, 3))