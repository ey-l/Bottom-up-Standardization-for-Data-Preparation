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
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head(4)
_input1.shape[0]
_input1.isna().sum()

def remove_ht(sir):
    idx = sir.find('%20')
    if idx == -1:
        return sir
    else:
        return sir[0:idx] + ' ' + sir[idx + 3:]
_input1['keyword'] = _input1['keyword'].fillna(_input1['keyword'].mode()[0], inplace=False)
_input1['keyword'] = _input1['keyword'].apply(remove_ht)
_input0['keyword'] = _input0['keyword'].fillna(_input0['keyword'].mode()[0], inplace=False)
_input0['keyword'] = _input0['keyword'].apply(remove_ht)

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
_input1['Number_Of_Hashtags'] = _input1['text'].apply(number_of_hashtags)
_input1['Pos_Sentiment'] = _input1['text'].apply(pos_sentiment)
_input1['Neg_Sentiment'] = _input1['text'].apply(neg_sentiment)
_input1['Number_Of_Words'] = _input1['text'].apply(number_of_words)
_input1['Exc_Marks'] = _input1['text'].apply(number_of_exclamation_marks)
_input1['Avg_Word_Length'] = _input1['text'].apply(average_word_length)
_input1['Has_Mention'] = _input1['text'].apply(contains_mentions)
_input1['Has_Disaster_Word'] = _input1['text'].apply(contains_disaster_tag)
_input0['Number_Of_Hashtags'] = _input0['text'].apply(number_of_hashtags)
_input0['Pos_Sentiment'] = _input0['text'].apply(pos_sentiment)
_input0['Neg_Sentiment'] = _input0['text'].apply(neg_sentiment)
_input0['Number_Of_Words'] = _input0['text'].apply(number_of_words)
_input0['Exc_Marks'] = _input0['text'].apply(number_of_exclamation_marks)
_input0['Avg_Word_Length'] = _input0['text'].apply(average_word_length)
_input0['Has_Mention'] = _input0['text'].apply(contains_mentions)
_input0['Has_Disaster_Word'] = _input0['text'].apply(contains_disaster_tag)
from sklearn.preprocessing import LabelEncoder
label_e = LabelEncoder()