import numpy as np
import pandas as pd
import os
import re
import emoji
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.head()

def missing_value_of_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percentage = round(total / data.shape[0] * 100, 2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])
missing_value_of_data(df)
df = df.dropna()

def count_values_in_column(data, feature):
    total = data.loc[:, feature].value_counts(dropna=False)
    percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])
count_values_in_column(df, 'target')

def unique_values_in_column(data, feature):
    unique_val = pd.Series(data.loc[:, feature].unique())
    return pd.concat([unique_val], axis=1, keys=['Unique Values'])
unique_values_in_column(df, 'target')

def duplicated_values_data(data):
    dup = []
    columns = data.columns
    for i in data.columns:
        dup.append(sum(data[i].duplicated()))
    return pd.concat([pd.Series(columns), pd.Series(dup)], axis=1, keys=['Columns', 'Duplicate count'])
duplicated_values_data(df)
df.describe()

def find_url(string):
    text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return ''.join(text)
sentence = 'I love spending time at https://www.kaggle.com/'
find_url(sentence)
df['url'] = df['text'].apply(lambda x: find_url(x))

def find_emoji(text):
    emo_text = emoji.demojize(text)
    line = re.findall('\\:(.*?)\\:', emo_text)
    return line
sentence = 'I love ⚽ very much 😁'
find_emoji(sentence)
df['emoji'] = df['text'].apply(lambda x: find_emoji(x))

def remove_emoji(text):
    emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    return emoji_pattern.sub('', text)
sentence = 'Its all about 😀 face'
print(sentence)
remove_emoji(sentence)
df['text'] = df['text'].apply(lambda x: remove_emoji(x))

def find_email(text):
    line = re.findall('[\\w\\.-]+@[\\w\\.-]+', str(text))
    return ','.join(line)
sentence = 'My gmail is abc99@gmail.com'
find_email(sentence)
df['email'] = df['text'].apply(lambda x: find_email(x))

def find_hash(text):
    line = re.findall('(?<=#)\\w+', text)
    return ' '.join(line)
sentence = '#Corona is trending now in the world'
find_hash(sentence)
df['hash'] = df['text'].apply(lambda x: find_hash(x))

def find_at(text):
    line = re.findall('(?<=@)\\w+', text)
    return ' '.join(line)
sentence = '@David,can you help me out'
find_at(sentence)
df['at_mention'] = df['text'].apply(lambda x: find_at(x))

def find_number(text):
    line = re.findall('[0-9]+', text)
    return ' '.join(line)
sentence = '2833047 people are affected by corona now'
find_number(sentence)
df['number'] = df['text'].apply(lambda x: find_number(x))

def find_phone_number(text):
    line = re.findall('\\b\\d{10}\\b', text)
    return ''.join(line)
find_phone_number('9998887776 is a phone number of Mark from 210,North Avenue')
df['phone_number'] = df['text'].apply(lambda x: find_phone_number(x))

def find_year(text):
    line = re.findall('\\b(19[40][0-9]|20[0-1][0-9]|2020)\\b', text)
    return line
sentence = 'India got independence on 1947.'
find_year(sentence)
df['year'] = df['text'].apply(lambda x: find_year(x))

def find_nonalp(text):
    line = re.findall('[^A-Za-z0-9 ]', text)
    return line
sentence = 'Twitter has lots of @ and # in posts.(general tweet)'
find_nonalp(sentence)
df['non_alp'] = df['text'].apply(lambda x: find_nonalp(x))

def find_punct(text):
    line = re.findall('[!"\\$%&\\\'()*+,\\-.\\/:;=#@?\\[\\\\\\]^_`{|}~]*', text)
    string = ''.join(line)
    return list(string)
example = 'Corona virus have kiled #24506 confirmed cases now.#Corona is un(tolerable)'
print(find_punct(example))
df['punctuation'] = df['text'].apply(lambda x: find_punct(x))

def stop_word_fn(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    non_stop_words = [w for w in word_tokens if not w in stop_words]
    stop_words = [w for w in word_tokens if w in stop_words]
    return stop_words
example_sent = 'This is a sample sentence, showing off the stop words filtration.'
stop_word_fn(example_sent)
df['stop_words'] = df['text'].apply(lambda x: stop_word_fn(x))

def ngrams_top(corpus, ngram_range, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """