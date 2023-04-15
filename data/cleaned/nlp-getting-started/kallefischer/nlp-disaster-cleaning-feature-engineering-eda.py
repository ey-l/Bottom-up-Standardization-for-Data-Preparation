import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re
import string
import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud

import advertools as adv
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
train.head()
test.drop('id', axis=1, inplace=True)
train.drop('id', axis=1, inplace=True)
print(f'Missing values in %:\n{train.isna().sum() / len(train) * 100}')
train['location'].dropna(inplace=True)
test['location'].dropna(inplace=True)
train.fillna('', inplace=True)
test.fillna('', inplace=True)
keylist = list(train['keyword'].unique())
keylist[0:10]

def keyclean(text):
    try:
        text = text.split('%20')
        text = ' '.join(text)
        return text
    except:
        return text
keylist = list(train['keyword'].unique())
train['keyword'] = train['keyword'].apply(keyclean)
test['keyword'] = test['keyword'].apply(keyclean)
keylist_2 = list(train['keyword'].unique())
print(f'Unique keywords before removing %20: {keylist[0:10]}')
print(f'Unique keywords after removing %20:  {keylist_2[0:10]}')

def count_chars(text):
    return len(text)

def count_words(text):
    return len(text.split())

def unique_words(text):
    return len(set(text.split()))

def hashtag_counts(text):
    return adv.extract_hashtags(text)['hashtag_counts'][0]

def mention_counts(text):
    return adv.extract_mentions(text)['mention_counts'][0]

def question_counts(text):
    return adv.extract_questions(text)['question_mark_counts'][0]

def url(text):
    count = 0
    text = text.split()
    for i in text:
        if i.startswith('http'):
            count = count + 1
    return count
train['chars'] = train['text'].apply(count_chars)
train['words'] = train['text'].apply(count_words)
train['unique_words'] = train['text'].apply(unique_words)
train['word_length'] = train['chars'] / train['words']
train['hashtag'] = train['text'].apply(hashtag_counts)
train['mention'] = train['text'].apply(mention_counts)
train['question'] = train['text'].apply(question_counts)
train['url'] = train['text'].apply(url)
test['chars'] = test['text'].apply(count_chars)
test['words'] = test['text'].apply(count_words)
test['unique_words'] = test['text'].apply(unique_words)
test['word_length'] = test['chars'] / test['words']
test['hashtag'] = test['text'].apply(hashtag_counts)
test['mention'] = test['text'].apply(mention_counts)
test['question'] = test['text'].apply(question_counts)
test['url'] = test['text'].apply(url)
train.head()

def cleaning(text):
    clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))
    return ' '.join(sentence)
pre_clean = train[['text', 'target']][101:111]
train['text'] = train['text'].apply(cleaning)
test['text'] = test['text'].apply(cleaning)
post_clean = train[['text', 'target']][101:111]
print(pre_clean, '\n')
print(post_clean)
target0 = train.loc[train['target'] == 0]
target1 = train.loc[train['target'] == 1]
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=[26, 16])
wc = WordCloud(background_color='white', width=1200, height=800).generate(' '.join(target0['keyword']))
ax1.imshow(wc)
ax1.axis('off')
ax1.set_title('Non-disaster keywords', fontsize=20)
wc2 = WordCloud(background_color='white', width=1200, height=800).generate(' '.join(target1['keyword']))
ax2.imshow(wc2)
ax2.axis('off')
ax2.set_title('Disaster keywords', fontsize=20)
sns.countplot(data=train, x='target')
(fig, axes) = plt.subplots(2, 2, figsize=(20, 10))
plt.subplots_adjust(hspace=0.4)
axes[0, 0].set_title('Character Distribution')
sns.histplot(train['chars'], bins=100, kde=True, alpha=0.7, ax=axes[0, 0])
axes[1, 0].set_title('Word Distribution')
sns.histplot(train['words'], bins=100, kde=True, alpha=0.7, ax=axes[1, 0])
axes[1, 1].set_title('Unique Word Distribution')
sns.histplot(train['unique_words'], bins=100, kde=True, alpha=0.7, ax=axes[1, 1])
axes[0, 1].set_title('Word Length Distribution')
sns.histplot(train['word_length'], bins=100, kde=True, alpha=0.7, ax=axes[0, 1])
(fig, axes) = plt.subplots(3, 2, figsize=(20, 10))
plt.subplots_adjust(hspace=0.4)
axes[0, 0].set_title('Word count of non-disaster')
sns.histplot(data=target0, x='words', ax=axes[0, 0])
axes[0, 1].set_title('Word count of disaster')
sns.histplot(data=target1, x='words', color='darkorange', ax=axes[0, 1])
axes[1, 0].set_title('Word length of non-disaster')
sns.histplot(data=target0, x='word_length', ax=axes[1, 0])
axes[1, 1].set_title('Word length of disaster')
sns.histplot(data=target1, x='word_length', color='darkorange', ax=axes[1, 1])
axes[2, 0].set_title('Characters of non-disaster')
sns.histplot(data=target0, x='chars', ax=axes[2, 0])
axes[2, 1].set_title('Characters of disaster')
sns.histplot(data=target1, x='chars', color='darkorange', ax=axes[2, 1])
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(20, 5))
plt.subplots_adjust(hspace=0.4)
ax1.set_title('Character Distribution')
sns.stripplot(data=train, x='target', y='chars', alpha=0.5, ax=ax1)
ax2.set_title('Word Distribution')
sns.stripplot(data=train, x='target', y='word_length', alpha=0.5, ax=ax2)
(fig, axes) = plt.subplots(2, 2, figsize=(20, 10))
plt.subplots_adjust(hspace=0.4)
axes[0, 0].set_title('Count of URL')
sns.countplot(data=train, x='url', palette='deep', ax=axes[0, 0])
axes[1, 0].set_title('Count of Hashtag')
sns.countplot(data=train, x='hashtag', palette='deep', ax=axes[1, 0])
axes[1, 1].set_title('Count of Mention')
sns.countplot(data=train, x='mention', palette='deep', ax=axes[1, 1])
axes[0, 1].set_title('Count of Question')
sns.countplot(data=train, x='question', palette='deep', ax=axes[0, 1])
url_ex0 = train.loc[train['url'] != 0]
question_ex0 = train.loc[train['question'] != 0]
hashtag_ex0 = train.loc[train['hashtag'] != 0]
mention_ex0 = train.loc[train['mention'] != 0]
(fig, axes) = plt.subplots(2, 2, figsize=(20, 10))
plt.subplots_adjust(hspace=0.4)
axes[0, 0].set_title('Count of URL')
sns.countplot(data=url_ex0, x='url', palette='deep', ax=axes[0, 0])
axes[1, 0].set_title('Count of Hashtag')
sns.countplot(data=hashtag_ex0, x='hashtag', palette='deep', ax=axes[1, 0])
axes[1, 1].set_title('Count of Mention')
sns.countplot(data=mention_ex0, x='mention', palette='deep', ax=axes[1, 1])
axes[0, 1].set_title('Count of Question')
sns.countplot(data=question_ex0, x='question', palette='deep', ax=axes[0, 1])
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(20, 5))
plt.subplots_adjust(wspace=0.25)
ax1.set_title('Most common non-disaster words')
target0['keyword'].value_counts(ascending=True)[-10:].plot.barh(ax=ax1)
ax2.set_title('Most common disaster words')
target1['keyword'].value_counts(ascending=True)[-10:].plot.barh(ax=ax2, color='darkorange')
train.head()