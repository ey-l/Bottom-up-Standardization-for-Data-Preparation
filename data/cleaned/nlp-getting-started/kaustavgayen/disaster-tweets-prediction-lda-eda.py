import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
import re

import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
train['keyword'].unique()
train['location'].unique()
train[train['keyword'] == 'accident'].head(7)
train[train['location'] == 'Canada'].head(7)
tweetLengthTrain = train['text'].str.len()
tweetLengthTest = test['text'].str.len()
plt.hist(tweetLengthTrain, bins=20, label='Train_Tweet')
plt.hist(tweetLengthTest, bins=20, label='Test_Tweet')
plt.legend()

train['target_mean'] = train.groupby('keyword')['target'].transform('mean')
plt.figure(figsize=(8, 72))
sns.countplot(y=train.sort_values('target', ascending=False)['keyword'], hue=train.sort_values('target', ascending=False)['target'])
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.title('Target Distribution in Keywords')

train_df = train
test_df = test
combine = train.append(test, ignore_index=True)
print('Shape of new Dataset:', combine.shape)
combine.tail()
text_tweet = combine['text']
text_tweet.tail()
df = pd.DataFrame(combine[['text']])
df.head(7)

def remove_punct(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
df['Tweet_punct'] = df['text'].apply(lambda x: remove_punct(x))
df.head(10)
df['Tweet_punct'].drop_duplicates(inplace=True)
df['Tweet_punct'].shape
combine['Tweet_punct'] = df['Tweet_punct']
combine.head()
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
df = combine[combine['Tweet_punct'].notna()]
df.head()

def tokenization(text):
    text = re.split('\\W+', text)
    return text
df['Tweet_tokenized'] = df['Tweet_punct'].apply(lambda x: tokenization(x.lower()))
df.head(10)
stemmer = PorterStemmer()
df['Tweet_tokenized'] = df['Tweet_tokenized'].apply(lambda x: [stemmer.stem(i) for i in x])
df.head()
for i in range(len(df['Tweet_tokenized'])):
    df['Tweet_tokenized'][i] = ' '.join(df['Tweet_tokenized'][i])
df['cleanedText'] = df['Tweet_tokenized']
df.head()
from wordcloud import WordCloud
allWords = ' '.join([text for text in df['cleanedText']])
wordcloud = WordCloud(background_color='white', width=800, height=500, random_state=25, max_font_size=110).generate(allWords)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

from textblob import TextBlob

def sentiment_polarity(text):
    tweet_text = TextBlob(str(text))
    sentiment_value = tweet_text.sentiment.polarity
    return sentiment_value
df['Tweet_Polarity'] = df['cleanedText'].apply(lambda x: sentiment_polarity(x.lower()))
df.head(10)

def sentiment_analysis(value):
    sentiment = ''
    if value < 0.0:
        sentiment = 'negative'
    elif value > 0.0:
        sentiment = 'positive'
    else:
        sentiment = 'neutral'
    return sentiment
df['Tweet_Sentiments'] = df['Tweet_Polarity'].apply(lambda x: sentiment_analysis(float(x)))
df[['Tweet_punct', 'Tweet_Polarity', 'Tweet_Sentiments']].head(10)
import seaborn as sns
ax = sns.countplot(x='Tweet_Sentiments', data=df)

def tokenization(text):
    text = re.split('\\W+', text)
    return text
df['Tweet_tokenized'] = df['cleanedText'].apply(lambda x: tokenization(x.lower()))
df.head(10)
from gensim.corpora import Dictionary
text_dict = Dictionary(df.Tweet_tokenized)
text_dict.token2id
tweets_bow = [text_dict.doc2bow(tweet) for tweet in df['Tweet_tokenized']]
tweets_bow
from gensim.models.ldamodel import LdaModel
k = 10
tweets_lda = LdaModel(tweets_bow, num_topics=k, id2word=text_dict, random_state=1, passes=10)
tweets_lda.show_topics()
train['word_count'] = train['text'].map(lambda x: len(str(x).split()))
test['word_count'] = test['text'].map(lambda x: len(str(x).split()))
train['unique_word_count'] = train['text'].map(lambda x: len(set(str(x).split())))
test['unique_word_count'] = test['text'].map(lambda x: len(set(str(x).split())))
train['stop_word_count'] = train['text'].map(lambda x: len([elt for elt in str(x).lower().split() if elt in stopwords.words('english')]))
test['stop_word_count'] = test['text'].map(lambda x: len([elt for elt in str(x).lower().split() if elt in stopwords.words('english')]))
train['url_count'] = train['text'].map(lambda x: len([w for w in str(x).lower().split() if 'http' or 'https' in w]))
test['url_count'] = test['text'].map(lambda x: len([w for w in str(x).lower().split() if 'http' or 'https' in w]))
train['mean_word_length'] = train['text'].map(lambda x: np.mean([len(word) for word in x.split()]))
test['mean_word_length'] = test['text'].map(lambda x: np.mean([len(word) for word in x.split()]))
train['char_count'] = train['text'].map(lambda x: len(str(x)))
test['char_count'] = test['text'].map(lambda x: len(str(x)))
train['punctuation_count'] = train['text'].map(lambda x: len([elt for elt in str(x) if elt in string.punctuation]))
test['punctuation_count'] = test['text'].map(lambda x: len([elt for elt in str(x) if elt in string.punctuation]))
train['hashtag_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
test['hashtag_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
train['mention_count'] = train['text'].map(lambda x: len([c for c in str(x) if c == '@']))
test['mention_count'] = test['text'].map(lambda x: len([c for c in str(x) if c == '@']))
train.head(4)
META_FEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'mean_word_length', 'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']
(fig, ax) = plt.subplots(nrows=len(META_FEATURES), ncols=2, figsize=(20, 50), dpi=100)
mask = train['target'] == 1
for (i, feature) in enumerate(META_FEATURES):
    sns.distplot(train[mask][feature], ax=ax[i, 0], label='Disaster', kde=False)
    sns.distplot(train[~mask][feature], ax=ax[i, 0], label='Not Disaster', kde=False)
    ax[i, 0].set_title('{} target distribution in trainning dataset'.format(feature), fontsize=13)
    sns.distplot(train[feature], ax=ax[i, 1], label='Train Dataset', kde=False)
    sns.distplot(test[feature], ax=ax[i, 1], label='Test Dataset', kde=False)
    ax[i, 1].set_title('{} training and test dataset distributions '.format(feature), fontsize=13)
    for j in range(2):
        ax[i, j].set_xlabel(' ')
        ax[i, j].tick_params(axis='x', labelsize=13)
        ax[i, j].tick_params(axis='y', labelsize=13)
        ax[i, j].legend()

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
train.groupby('target').count()['id'].plot(kind='pie', labels=['Not Disaster', 'Disaster'], autopct='%1.1f pourcents', ax=ax[0])
sns.countplot(x=train['target'], hue=train['target'], ax=ax[1])
ax[1].set_xticklabels(['Non Disaster', 'Disaster'])
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
ax[0].set_ylabel('')
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[0].set_title('Target distribution in training set', fontsize=13)
ax[1].set_title('Target count in training set', fontsize=13)

def gen_n_grams(text, n_grams=1):
    """ This function allow to extract the n_gram in the introduced text.
    
      @param text(str): the text that the function, will use to extract features (n_grams).
      @param n_grams(int): the length of n_gram, that we will use.
      @return ngrams(list): list of the ngrams in the intriduced text.
    """
    tokens = [token for token in str(text).lower().split() if token not in stopwords.words('english')]
    ngrams = zip(*[tokens[i:] for i in range(n_grams)])
    return [' '.join(gram) for gram in ngrams]

def gen_df_ngrams(n_grams=1):
    """ This function, allow to generate dataframes for n_grams in disaster tweets and non 
        disaster tweet
    """
    mask = train['target'] == 1
    disaster_unigrams = defaultdict(int)
    non_disaster_unigrams = defaultdict(int)
    for tweet in train.loc[mask, 'text'].values:
        for gram in gen_n_grams(tweet, n_grams=n_grams):
            disaster_unigrams[gram] += 1
    for tweet in train.loc[~mask, 'text'].values:
        for gram in gen_n_grams(tweet, n_grams=n_grams):
            non_disaster_unigrams[gram] += 1
    df_disaster_n_grams = pd.DataFrame(sorted(disaster_unigrams.items(), reverse=True, key=lambda item: item[1]))
    df_non_disaster_n_grams = pd.DataFrame(sorted(non_disaster_unigrams.items(), reverse=True, key=lambda item: item[1]))
    return (df_disaster_n_grams, df_non_disaster_n_grams)

def plot_ngrams(df_disaster_n_grams, df_non_disaster_unigrams, N=100, n_grams=1):
    """This function,allow to plot the top most n_grams in disaster tweet and non disaser tweet.
    """
    (fig, ax) = plt.subplots(1, 2, figsize=(18, 50))
    sns.barplot(y=df_disaster_n_grams[0].values[:N], x=df_disaster_n_grams[1].values[:N], ax=ax[0], color='red')
    for i in range(2):
        ax[i].tick_params(axis='x', labelsize=15)
        ax[i].tick_params(axis='y', labelsize=15)
        ax[i].set_xlabel('Occurences')
        ax[i].spines['right'].set_visible(False)
    sns.barplot(y=df_non_disaster_unigrams[0].values[:N], x=df_non_disaster_unigrams[1].values[:N], ax=ax[1], color='green')
    ax[0].set_title('Top most {} {}_grams for disaster tweets'.format(N, n_grams), size=15)
    ax[1].set_title('Top most {} {}_grams for non disaster tweets'.format(N, n_grams), size=15)
(df_disaster_unigrams, df_non_disaster_unigrams) = gen_df_ngrams()
plot_ngrams(df_disaster_unigrams, df_non_disaster_unigrams)
(df_disaster_unigrams, df_non_disaster_unigrams) = gen_df_ngrams(n_grams=2)
plot_ngrams(df_disaster_unigrams, df_non_disaster_unigrams, n_grams=2)
(df_disaster_unigrams, df_non_disaster_unigrams) = gen_df_ngrams(n_grams=3)
plot_ngrams(df_disaster_unigrams, df_non_disaster_unigrams, n_grams=3)

def process_text(text):
    """
    Removes punctuations(if any), stopwords and returns a list words
    """
    rm_pun = [char for char in text if char not in string.punctuation]
    rm_pun = ''.join(rm_pun)
    return [word for word in rm_pun.split() if word.lower() not in stopwords.words('english')]