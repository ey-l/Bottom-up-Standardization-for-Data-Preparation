import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_colwidth', None)
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
test.head()
df_train = train.copy()
df_test = test.copy()
df_train.head()
df_test.head()
df_train.info()
df_train.describe()
df_train.describe(include='object')
df_train.isnull().sum() / df_train.shape[0] * 100
df_train.loc[df_train.keyword.isnull() == True, :].head()
df_train.loc[df_train.location.isnull() == True, :].head()
df_test.isnull().sum() / df_test.shape[0] * 100
df_test.loc[df_test.location.isnull() == True, :].head()
df_test.loc[df_test.keyword.isnull() == True, :].head()
df_train.shape
df_test.shape
cols = df_train.columns
for i in range(0, len(cols)):
    print('Column :', cols[i].upper())
    print(df_train[cols[i]].value_counts(dropna=True)[:5])
    print('********************************************')
cols = df_test.columns
for i in range(0, len(cols)):
    print('Column :', cols[i].upper())
    print(df_test[cols[i]].value_counts(dropna=True)[:5])
    print('********************************************')
sns.set_style()
sns.countplot(data=df_train, x=df_train['target'], palette='rocket')
plt.title('Distribution of Target variable')
plt.xlabel('Target')
plt.ylabel('Count of Target')

df_train['target'].value_counts()
sns.distplot(df_train['target'])

sns.barplot(y=df_train['keyword'].value_counts()[:20].index, x=df_train['keyword'].value_counts()[:20])
df_train.loc[df_train['text'].str.contains('disaster', na=False, case=False)].target.value_counts()
sns.barplot(y=df_train['location'].value_counts()[:10].index, x=df_train['location'].value_counts()[:10], orient='h')
df_train['location'].replace({'United States': 'USA', 'New York': 'USA', 'London': 'UK', 'Los Angeles, CA': 'USA', 'Washington, D.C.': 'USA', 'California': 'USA', 'Chicago, IL': 'USA', 'Chicago': 'USA', 'New York, NY': 'USA', 'California, USA': 'USA', 'FLorida': 'USA', 'Nigeria': 'Africa', 'Kenya': 'Africa', 'Everywhere': 'Worldwide', 'San Francisco': 'USA', 'Florida': 'USA', 'United Kingdom': 'UK', 'Los Angeles': 'USA', 'Toronto': 'Canada', 'San Francisco, CA': 'USA', 'NYC': 'USA', 'Seattle': 'USA', 'Earth': 'Worldwide', 'Ireland': 'UK', 'London, England': 'UK', 'New York City': 'USA', 'Texas': 'USA', 'London, UK': 'UK', 'Atlanta, GA': 'USA', 'Mumbai': 'India', 'Sao Paulo, Brazil': 'Brazil'}, inplace=True)
sns.barplot(y=df_train['location'].value_counts()[:10].index, x=df_train['location'].value_counts()[:10], orient='h')
df_train.loc[df_train['location'] == 'ss', :]
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5))
tweet_len = df_train[df_train['target'] == 1]['text'].str.len()
ax1.hist(tweet_len, color='red')
ax1.set_title('disaster tweets')
tweet_len = df_train[df_train['target'] == 0]['text'].str.len()
ax2.hist(tweet_len, color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')

(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5))
tweet_len = df_train[df_train['target'] == 1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len, color='red')
ax1.set_title('disaster tweets')
tweet_len = df_train[df_train['target'] == 0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len, color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Words in a tweet')

df_train['text'][:25]
import re
import string

def preprocessing(text):
    """
    Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.
    """
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    text = re.sub('[#|@|!|$|%|\x89|^|&|*|(|)|[|{|[|\\]]', '', text)
    text = re.sub('im', 'i am', text)
    text = re.sub('û', 'u', text)
    text = text.strip()
    return text

def remove_emoji(text):
    emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    return emoji_pattern.sub('', text)
df_train['text'] = df_train['text'].apply(lambda x: preprocessing(x))
df_train['text'] = df_train['text'].apply(lambda x: remove_emoji(x))
df_test['text'] = df_test['text'].apply(lambda x: preprocessing(x))
df_test['text'] = df_test['text'].apply(lambda x: remove_emoji(x))
df_train['text'][:5]
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stopwords(string):
    word_list = [word for word in string.split()]
    stopwords_list = list(stopwords.words('english'))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    return ' '.join(word_list)
df_train['text'] = list(map(lambda x: remove_stopwords(x), df_train['text']))
df_test['text'] = list(map(lambda x: remove_stopwords(x), df_test['text']))
df_train.head()
non_disaster_tweets = df_train[df_train['target'] == 0]['text']
non_disaster_tweets.values[1]
disaster_tweets = df_train[df_train['target'] == 1]['text']
disaster_tweets.values[1]
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=[26, 8])
wordcloud1 = WordCloud(width=600, height=400).generate(' '.join(disaster_tweets))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Disaster Tweets', fontsize=30)
wordcloud2 = WordCloud(width=600, height=400).generate(' '.join(non_disaster_tweets))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Non Disaster Tweets', fontsize=30)
df_train['text'][66]