import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.metrics import f1_score as f1
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objs as go
import plotly.express as ex
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
df_test = pd.read_csv('data/input/nlp-getting-started/test.csv')
df_train.head(4)
df_train.shape[0]
df_train.isna().sum()

def remove_ht(sir):
    idx = sir.find('%20')
    if idx == -1:
        return sir
    else:
        return sir[0:idx] + ' ' + sir[idx + 3:]
df_train['keyword'].fillna(df_train['keyword'].mode()[0], inplace=True)
df_train['keyword'] = df_train['keyword'].apply(remove_ht)
df_test['keyword'].fillna(df_test['keyword'].mode()[0], inplace=True)
df_test['keyword'] = df_test['keyword'].apply(remove_ht)

def number_of_hashtags(sir):
    splited = sir.split(' ')
    ht = 0
    for word in splited:
        if len(word) > 1 and word[0] == '#':
            ht += 1
    return ht
sid = SentimentIntensityAnalyzer()

def pos_sentiment(sir):
    r = sid.polarity_scores(sir)
    return r['pos']

def neg_sentiment(sir):
    r = sid.polarity_scores(sir)
    return r['neg']

def number_of_words(sir):
    splited = sir.split(' ')
    words = 0
    for word in splited:
        if len(word) > 1 and word[0] != '#':
            words += 1
    return words

def number_of_exclamation_marks(sir):
    ex = 0
    for char in sir:
        if char == '!':
            ex += 1
    return ex

def average_word_length(sir):
    splited = sir.split(' ')
    no_hash = [word for word in splited if '#' not in word]
    length = 0
    for word in no_hash:
        length += len(word)
    return length / len(no_hash)

def contains_mentions(sir):
    splited = sir.split(' ')
    for word in splited:
        if '@' in word:
            return 1
    return 0
disasters = ['fire', 'storm', 'flood', 'tornado', 'earthquake', 'volcano', 'hurricane', 'tornado', 'cyclone', 'famine', 'epidemic', 'war', 'dead', 'collapse', 'crash', 'hostages', 'terror']

def contains_disaster_tag(sir):
    splited = sir.split(' ')
    for word in splited:
        if word.lower() in disasters:
            return 1
    return 0
df_train['Number_Of_Hashtags'] = df_train['text'].apply(number_of_hashtags)
df_train['Pos_Sentiment'] = df_train['text'].apply(pos_sentiment)
df_train['Neg_Sentiment'] = df_train['text'].apply(neg_sentiment)
df_train['Number_Of_Words'] = df_train['text'].apply(number_of_words)
df_train['Exc_Marks'] = df_train['text'].apply(number_of_exclamation_marks)
df_train['Avg_Word_Length'] = df_train['text'].apply(average_word_length)
df_train['Has_Mention'] = df_train['text'].apply(contains_mentions)
df_train['Has_Disaster_Word'] = df_train['text'].apply(contains_disaster_tag)
df_test['Number_Of_Hashtags'] = df_test['text'].apply(number_of_hashtags)
df_test['Pos_Sentiment'] = df_test['text'].apply(pos_sentiment)
df_test['Neg_Sentiment'] = df_test['text'].apply(neg_sentiment)
df_test['Number_Of_Words'] = df_test['text'].apply(number_of_words)
df_test['Exc_Marks'] = df_test['text'].apply(number_of_exclamation_marks)
df_test['Avg_Word_Length'] = df_test['text'].apply(average_word_length)
df_test['Has_Mention'] = df_test['text'].apply(contains_mentions)
df_test['Has_Disaster_Word'] = df_test['text'].apply(contains_disaster_tag)
from sklearn.preprocessing import LabelEncoder
label_e = LabelEncoder()