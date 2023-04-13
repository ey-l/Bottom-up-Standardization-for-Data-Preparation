import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
import string
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1.head()
_input1 = _input1.rename(columns={'target': 'real_tweet'}, inplace=False)
_input1.info()
_input1['real_tweet'].value_counts(normalize=True)
plt.figure(figsize=(15, 10))
sns.countplot(data=_input1, x='real_tweet', palette='twilight', saturation=1)
real_tweets = _input1[_input1['real_tweet'] == 1]['text']
fake_tweets = _input1[_input1['real_tweet'] == 0]['text']
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
_input1['len_text'] = _input1['text'].apply(len)
_input0['len_text'] = _input0['text'].apply(len)
_input1.groupby(['real_tweet'])['len_text'].describe().T
plt.figure(figsize=(15, 10))
sns.boxplot(data=_input1, x='len_text', y='real_tweet', orient='h', palette='YlGnBu')

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
_input1['check_digits'] = _input1['text'].apply(lambda x: check_digits(x))
_input1['check_punctation'] = _input1['text'].apply(lambda x: check_punctation(x))
_input1['count_punctation'] = _input1['text'].apply(lambda x: count_punctation(x))
_input1['count_digits'] = _input1['text'].apply(lambda x: count_digits(x))
_input0['check_digits'] = _input0['text'].apply(lambda x: check_digits(x))
_input0['check_punctation'] = _input0['text'].apply(lambda x: check_punctation(x))
_input0['count_punctation'] = _input0['text'].apply(lambda x: count_punctation(x))
_input0['count_digits'] = _input0['text'].apply(lambda x: count_digits(x))
_input1.groupby(['real_tweet'])['count_punctation'].describe().T
_input1.groupby(['real_tweet'])['count_digits'].mean()
(fig, ax) = plt.subplots(1, 2, figsize=(15, 10))
sns.boxenplot(data=_input1, x='count_digits', y='real_tweet', orient='h', ax=ax[0], palette='cubehelix')
sns.boxenplot(data=_input1, x='count_punctation', y='real_tweet', orient='h', ax=ax[1], palette='cubehelix')
(fig, ax) = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=_input1, x='check_digits', hue='real_tweet', ax=ax[0], palette='Set1')
sns.countplot(data=_input1, x='check_punctation', hue='real_tweet', ax=ax[1], palette='Set1')
_input1['check_http'] = _input1['text'].str.contains('http')
_input1['check_hash'] = _input1['text'].str.contains('#')
_input0['check_http'] = _input0['text'].str.contains('http')
_input0['check_hash'] = _input0['text'].str.contains('#')
(fig, ax) = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=_input1, x='check_http', hue='real_tweet', ax=ax[0], palette='Set2')
sns.countplot(data=_input1, x='check_hash', hue='real_tweet', ax=ax[1], palette='Set2')
_input1.head()

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
check_text_real = _input1[_input1['real_tweet'] == 1]['text'].head(5)
check_text_fake = _input1[_input1['real_tweet'] == 0]['text'].head(5)
for (i, text) in enumerate(check_text_real):
    print(f'Clean [{i + 1}]:', clean_text(text), end='\n\n')
    print(f'Real [{i + 1}]:', text, end='\n\n')
    print()
for (i, text) in enumerate(check_text_fake):
    print(f'Clean [{i + 1}]:', clean_text(text), end='\n\n')
    print(f'Real [{i + 1}]:', text, end='\n\n')
    print()
_input1['clean_text'] = _input1['text'].apply(clean_text)
_input0['clean_text'] = _input0['text'].apply(clean_text)
_input1.head().T
_input1['keyword'].isnull().sum()
_input1['keyword'] = _input1['keyword'].fillna('Empty')
_input0['keyword'] = _input0['keyword'].fillna('Empty')
_input1['keyword'].unique()
_input1['keyword'].nunique()
_input1[_input1['real_tweet'] == 1]['keyword'].value_counts()
plt.figure(figsize=(6, 75), dpi=100)
sns.countplot(data=_input1, y='keyword', hue='real_tweet', order=_input1[_input1['real_tweet'] == 1]['keyword'].value_counts().index, palette='bone')
_input1['location'].isnull().sum()
print('missing values :', round(_input1['location'].isnull().sum() / _input1.shape[0], 3), '%')
_input1['location'].nunique()
top_location = _input1['location'].value_counts(normalize=True).head(50).index
plt.figure(figsize=(15, 10))
sns.countplot(data=_input1[_input1['location'].isin(top_location)], x='location', hue='real_tweet', palette='Pastel2_r')
plt.xticks(rotation=80)
clean_train = _input1.drop(['count_digits', 'count_punctation', 'id', 'location', 'text'], axis=1)
clean_test = _input0.drop(['count_digits', 'count_punctation', 'id', 'location', 'text'], axis=1)
(fig, ax) = plt.subplots(1, 2, figsize=(15, 5))
ax[0].title.set_text('Before\n')
sns.heatmap(_input1.corr(), annot=True, ax=ax[0], fmt='.1g', cmap='Greens', linewidths=2, linecolor='black')
ax[1].title.set_text('After\n')
sns.heatmap(clean_train.corr(), annot=True, ax=ax[1], fmt='.1g', cmap='Greens', linewidths=2, linecolor='black')
clean_train.head()
clean_train['check_http'] = clean_train['check_http'].apply(lambda x: 1 if x else 0)
clean_train['check_hash'] = clean_train['check_hash'].apply(lambda x: 1 if x else 0)
clean_test['check_http'] = clean_test['check_http'].apply(lambda x: 1 if x else 0)
clean_test['check_hash'] = clean_test['check_hash'].apply(lambda x: 1 if x else 0)
from sklearn.preprocessing import MinMaxScaler
scal = MinMaxScaler()