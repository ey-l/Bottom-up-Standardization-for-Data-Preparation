import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
import string
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
output_example = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
train.head()
train.rename(columns={'target': 'real_tweet'}, inplace=True)
train.info()
train['real_tweet'].value_counts(normalize=True)
plt.figure(figsize=(15, 10))
sns.countplot(data=train, x='real_tweet', palette='twilight', saturation=1)
real_tweets = train[train['real_tweet'] == 1]['text']
fake_tweets = train[train['real_tweet'] == 0]['text']
random_real_tweets = real_tweets.sample(5).values
random_fake_tweets = fake_tweets.sample(5).values
print('Real tweets:\n')
for (i, tweet) in enumerate(random_real_tweets):
    print('[{0}] {1}\n'.format(i + 1, tweet))
    if i == 5:
        break
print('-' * 90)
print('Fake tweets:\n')
for (i, tweet) in enumerate(random_fake_tweets):
    print('[{0}] {1}\n'.format(i + 1, tweet))
    if i == 5:
        break
train['len_text'] = train['text'].apply(len)
test['len_text'] = test['text'].apply(len)
train.groupby(['real_tweet'])['len_text'].describe().T
plt.figure(figsize=(15, 10))
sns.boxplot(data=train, x='len_text', y='real_tweet', orient='h', palette='YlGnBu')

def check_digits(text):
    if [char for char in text if char in string.digits]:
        return 1
    else:
        return 0

def check_punctation(text):
    if [char for char in text if char in string.punctuation]:
        return 1
    else:
        return 0

def count_punctation(text):
    count = 0
    for char in text:
        if char in string.punctuation:
            count += 1
    return count

def count_digits(text):
    count = 0
    for char in text:
        if char in string.digits:
            count += 1
    return count
train['check_digits'] = train['text'].apply(lambda x: check_digits(x))
train['check_punctation'] = train['text'].apply(lambda x: check_punctation(x))
train['count_punctation'] = train['text'].apply(lambda x: count_punctation(x))
train['count_digits'] = train['text'].apply(lambda x: count_digits(x))
test['check_digits'] = test['text'].apply(lambda x: check_digits(x))
test['check_punctation'] = test['text'].apply(lambda x: check_punctation(x))
test['count_punctation'] = test['text'].apply(lambda x: count_punctation(x))
test['count_digits'] = test['text'].apply(lambda x: count_digits(x))
train.groupby(['real_tweet'])['count_punctation'].describe().T
train.groupby(['real_tweet'])['count_digits'].mean()
(fig, ax) = plt.subplots(1, 2, figsize=(15, 10))
sns.boxenplot(data=train, x='count_digits', y='real_tweet', orient='h', ax=ax[0], palette='cubehelix')
sns.boxenplot(data=train, x='count_punctation', y='real_tweet', orient='h', ax=ax[1], palette='cubehelix')
(fig, ax) = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=train, x='check_digits', hue='real_tweet', ax=ax[0], palette='Set1')
sns.countplot(data=train, x='check_punctation', hue='real_tweet', ax=ax[1], palette='Set1')
train['check_http'] = train['text'].str.contains('http')
train['check_hash'] = train['text'].str.contains('#')
test['check_http'] = test['text'].str.contains('http')
test['check_hash'] = test['text'].str.contains('#')
(fig, ax) = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=train, x='check_http', hue='real_tweet', ax=ax[0], palette='Set2')
sns.countplot(data=train, x='check_hash', hue='real_tweet', ax=ax[1], palette='Set2')
train.head()

def clean_text(text):
    stop_words = stopwords.words('english')
    stop_words.append("i'm")
    text = text.lower()
    text = ' '.join([char for char in text.split() if char not in stop_words])
    text = ' '.join(['' if 'http' in char else char for char in text.split()])
    text = ''.join([char + ' ' if char in string.punctuation else char for char in text])
    text = ''.join([char for char in text if char in string.ascii_lowercase or char in ' ' or char in string.digits])
    text = ' '.join(text.split())
    return text
check_text_real = train[train['real_tweet'] == 1]['text'].head(5)
check_text_fake = train[train['real_tweet'] == 0]['text'].head(5)
for (i, text) in enumerate(check_text_real):
    print(f'Clean [{i + 1}]:', clean_text(text), end='\n\n')
    print(f'Real [{i + 1}]:', text, end='\n\n')
    print()
for (i, text) in enumerate(check_text_fake):
    print(f'Clean [{i + 1}]:', clean_text(text), end='\n\n')
    print(f'Real [{i + 1}]:', text, end='\n\n')
    print()
train['clean_text'] = train['text'].apply(clean_text)
test['clean_text'] = test['text'].apply(clean_text)
train.head().T
train['keyword'].isnull().sum()
train['keyword'] = train['keyword'].fillna('Empty')
test['keyword'] = test['keyword'].fillna('Empty')
train['keyword'].unique()
train['keyword'].nunique()
train[train['real_tweet'] == 1]['keyword'].value_counts()
plt.figure(figsize=(6, 75), dpi=100)
sns.countplot(data=train, y='keyword', hue='real_tweet', order=train[train['real_tweet'] == 1]['keyword'].value_counts().index, palette='bone')
train['location'].isnull().sum()
print('missing values :', round(train['location'].isnull().sum() / train.shape[0], 3), '%')
train['location'].nunique()
top_location = train['location'].value_counts(normalize=True).head(50).index
plt.figure(figsize=(15, 10))
sns.countplot(data=train[train['location'].isin(top_location)], x='location', hue='real_tweet', palette='Pastel2_r')
plt.xticks(rotation=80)
clean_train = train.drop(['count_digits', 'count_punctation', 'id', 'location', 'text'], axis=1)
clean_test = test.drop(['count_digits', 'count_punctation', 'id', 'location', 'text'], axis=1)
(fig, ax) = plt.subplots(1, 2, figsize=(15, 5))
ax[0].title.set_text('Before\n')
sns.heatmap(train.corr(), annot=True, ax=ax[0], fmt='.1g', cmap='Greens', linewidths=2, linecolor='black')
ax[1].title.set_text('After\n')
sns.heatmap(clean_train.corr(), annot=True, ax=ax[1], fmt='.1g', cmap='Greens', linewidths=2, linecolor='black')
clean_train.head()
clean_train['check_http'] = clean_train['check_http'].apply(lambda x: 1 if x else 0)
clean_train['check_hash'] = clean_train['check_hash'].apply(lambda x: 1 if x else 0)
clean_test['check_http'] = clean_test['check_http'].apply(lambda x: 1 if x else 0)
clean_test['check_hash'] = clean_test['check_hash'].apply(lambda x: 1 if x else 0)
from sklearn.preprocessing import MinMaxScaler
scal = MinMaxScaler()