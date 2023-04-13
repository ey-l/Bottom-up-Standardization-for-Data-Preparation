import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1
_input1.info()
_input1.isnull().sum()
_input1 = _input1.dropna(axis=1)
_input1
_input1.columns
tweets = _input1.drop(['id', 'target'], axis=1)
tweets.head()
print(tweets['text'].iloc[0], '\n')
print(tweets['text'].iloc[1], '\n')
print(tweets['text'].iloc[2], '\n')
print(tweets['text'].iloc[3], '\n')
print(tweets['text'].iloc[4], '\n')
tweets.info()

def data_processing(text):
    text = text.lower()
    text = re.sub('https\\S+|www\\S+https\\S+', '', text, flags=re.MULTILINE)
    text = re.sub('\\@w+|\\#', '', text)
    text = re.sub('[^\\w\\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return ' '.join(filtered_text)
tweets.text = tweets['text'].apply(data_processing)
tweets = tweets.drop_duplicates('text')

def polarity(text):
    return TextBlob(text).sentiment.polarity
tweets['polarity'] = tweets['text'].apply(polarity)
stemmer = PorterStemmer()

def stemming(data):
    data = [stemmer.stem(word) for word in data]
    return data
tweets['SplittedText'] = tweets['text'].apply(lambda x: stemming(x))
tweets.head()
print(tweets['text'].iloc[0], '\n')
print(tweets['text'].iloc[1], '\n')
print(tweets['text'].iloc[2], '\n')
print(tweets['text'].iloc[3], '\n')
print(tweets['text'].iloc[4], '\n')
tweets.info()
tweets.head()

def sentiment(label):
    if label < 0:
        return 'Negative'
    elif label == 0:
        return 'Neutral'
    elif label > 0:
        return 'Positive'
tweets['sentiment'] = tweets['polarity'].apply(sentiment)
tweets.head()
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='sentiment', data=tweets)
fig = plt.figure(figsize=(7, 7))
colors = ('yellowgreen', 'gold', 'red')
wp = {'linewidth': 2, 'edgecolor': 'black'}
tags = tweets['sentiment'].value_counts()
explode = (0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='')
plt.title('Sentiments')
positive_tweets = tweets[tweets.sentiment == 'Positive']
positive_tweets = positive_tweets.sort_values(['polarity'], ascending=False)
positive_tweets.head()
negative_tweets = tweets[tweets.sentiment == 'Negative']
negative_tweets = negative_tweets.sort_values(['polarity'], ascending=False)
negative_tweets.head()
neutral_tweets = tweets[tweets.sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['polarity'], ascending=False)
neutral_tweets.head()
text = ' '.join([word for word in negative_tweets['text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words used in Negative Tweets', fontsize=19)
text = ' '.join([word for word in neutral_tweets['text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words used in Neutral Tweets', fontsize=19)
text = ' '.join([word for word in positive_tweets['text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words used in Positive Tweets', fontsize=19)