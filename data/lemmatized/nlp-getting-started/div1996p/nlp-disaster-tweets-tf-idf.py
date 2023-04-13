import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
stopword_ls = stopwords.words('english')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()

def text_to_lower_case(column, df):
    df[column] = df[column].str.lower()
    return df
_input1 = text_to_lower_case('text', _input1)
_input0 = text_to_lower_case('text', _input0)

def remove_punctuation(column, df):
    df[column] = df[column].str.replace('[^\\w\\s]', '')
    return df
_input1 = remove_punctuation('text', _input1)
_input0 = remove_punctuation('text', _input0)

def remove_numeric(column, df):
    df[column] = df[column].str.replace('\\d+', '')
    return df
_input1 = remove_numeric('text', _input1)
_input0 = remove_numeric('text', _input0)
print(_input1.text.isnull().sum())
print(_input0.text.isnull().sum())

def remove_stopwords(column, df, stopword):
    df[column] = df[column].apply(lambda x: ' '.join((word for word in x.split() if word not in stopword)))
    return df
_input1 = remove_stopwords('text', _input1, stopword_ls)
_input0 = remove_stopwords('text', _input0, stopword_ls)
_input1.head()
from wordcloud import WordCloud
from matplotlib import pyplot as plt
text = ' '.join((title for title in _input1.text))
word_cloud = WordCloud(collocations=False, background_color='white').generate(text)
plt.imshow(word_cloud, interpolation='bilinear')
from wordcloud import WordCloud
from matplotlib import pyplot as plt
text_ts = ' '.join((title for title in _input0.text))
word_cloud = WordCloud(collocations=False, background_color='white').generate(text_ts)
plt.imshow(word_cloud, interpolation='bilinear')
tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')