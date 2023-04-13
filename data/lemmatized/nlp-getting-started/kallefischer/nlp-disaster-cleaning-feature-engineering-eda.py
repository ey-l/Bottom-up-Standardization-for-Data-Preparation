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
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input0 = _input0.drop('id', axis=1, inplace=False)
_input1 = _input1.drop('id', axis=1, inplace=False)
print(f'Missing values in %:\n{_input1.isna().sum() / len(_input1) * 100}')
_input1['location'] = _input1['location'].dropna(inplace=False)
_input0['location'] = _input0['location'].dropna(inplace=False)
_input1 = _input1.fillna('', inplace=False)
_input0 = _input0.fillna('', inplace=False)
keylist = list(_input1['keyword'].unique())
keylist[0:10]

def keyclean(text):
    try:
        text = text.split('%20')
        text = ' '.join(text)
        return text
    except:
        return text
keylist = list(_input1['keyword'].unique())
_input1['keyword'] = _input1['keyword'].apply(keyclean)
_input0['keyword'] = _input0['keyword'].apply(keyclean)
keylist_2 = list(_input1['keyword'].unique())
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
_input1['chars'] = _input1['text'].apply(count_chars)
_input1['words'] = _input1['text'].apply(count_words)
_input1['unique_words'] = _input1['text'].apply(unique_words)
_input1['word_length'] = _input1['chars'] / _input1['words']
_input1['hashtag'] = _input1['text'].apply(hashtag_counts)
_input1['mention'] = _input1['text'].apply(mention_counts)
_input1['question'] = _input1['text'].apply(question_counts)
_input1['url'] = _input1['text'].apply(url)
_input0['chars'] = _input0['text'].apply(count_chars)
_input0['words'] = _input0['text'].apply(count_words)
_input0['unique_words'] = _input0['text'].apply(unique_words)
_input0['word_length'] = _input0['chars'] / _input0['words']
_input0['hashtag'] = _input0['text'].apply(hashtag_counts)
_input0['mention'] = _input0['text'].apply(mention_counts)
_input0['question'] = _input0['text'].apply(question_counts)
_input0['url'] = _input0['text'].apply(url)
_input1.head()

def cleaning(text):
    clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))
    return ' '.join(sentence)
pre_clean = _input1[['text', 'target']][101:111]
_input1['text'] = _input1['text'].apply(cleaning)
_input0['text'] = _input0['text'].apply(cleaning)
post_clean = _input1[['text', 'target']][101:111]
print(pre_clean, '\n')
print(post_clean)
target0 = _input1.loc[_input1['target'] == 0]
target1 = _input1.loc[_input1['target'] == 1]
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=[26, 16])
wc = WordCloud(background_color='white', width=1200, height=800).generate(' '.join(target0['keyword']))
ax1.imshow(wc)
ax1.axis('off')
ax1.set_title('Non-disaster keywords', fontsize=20)
wc2 = WordCloud(background_color='white', width=1200, height=800).generate(' '.join(target1['keyword']))
ax2.imshow(wc2)
ax2.axis('off')
ax2.set_title('Disaster keywords', fontsize=20)
sns.countplot(data=_input1, x='target')
(fig, axes) = plt.subplots(2, 2, figsize=(20, 10))
plt.subplots_adjust(hspace=0.4)
axes[0, 0].set_title('Character Distribution')
sns.histplot(_input1['chars'], bins=100, kde=True, alpha=0.7, ax=axes[0, 0])
axes[1, 0].set_title('Word Distribution')
sns.histplot(_input1['words'], bins=100, kde=True, alpha=0.7, ax=axes[1, 0])
axes[1, 1].set_title('Unique Word Distribution')
sns.histplot(_input1['unique_words'], bins=100, kde=True, alpha=0.7, ax=axes[1, 1])
axes[0, 1].set_title('Word Length Distribution')
sns.histplot(_input1['word_length'], bins=100, kde=True, alpha=0.7, ax=axes[0, 1])
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
sns.stripplot(data=_input1, x='target', y='chars', alpha=0.5, ax=ax1)
ax2.set_title('Word Distribution')
sns.stripplot(data=_input1, x='target', y='word_length', alpha=0.5, ax=ax2)
(fig, axes) = plt.subplots(2, 2, figsize=(20, 10))
plt.subplots_adjust(hspace=0.4)
axes[0, 0].set_title('Count of URL')
sns.countplot(data=_input1, x='url', palette='deep', ax=axes[0, 0])
axes[1, 0].set_title('Count of Hashtag')
sns.countplot(data=_input1, x='hashtag', palette='deep', ax=axes[1, 0])
axes[1, 1].set_title('Count of Mention')
sns.countplot(data=_input1, x='mention', palette='deep', ax=axes[1, 1])
axes[0, 1].set_title('Count of Question')
sns.countplot(data=_input1, x='question', palette='deep', ax=axes[0, 1])
url_ex0 = _input1.loc[_input1['url'] != 0]
question_ex0 = _input1.loc[_input1['question'] != 0]
hashtag_ex0 = _input1.loc[_input1['hashtag'] != 0]
mention_ex0 = _input1.loc[_input1['mention'] != 0]
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
_input1.head()