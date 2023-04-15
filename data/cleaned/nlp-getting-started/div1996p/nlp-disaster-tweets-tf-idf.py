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
train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_data.head()

def text_to_lower_case(column, df):
    df[column] = df[column].str.lower()
    return df
train_data = text_to_lower_case('text', train_data)
test_data = text_to_lower_case('text', test_data)

def remove_punctuation(column, df):
    df[column] = df[column].str.replace('[^\\w\\s]', '')
    return df
train_data = remove_punctuation('text', train_data)
test_data = remove_punctuation('text', test_data)

def remove_numeric(column, df):
    df[column] = df[column].str.replace('\\d+', '')
    return df
train_data = remove_numeric('text', train_data)
test_data = remove_numeric('text', test_data)
print(train_data.text.isnull().sum())
print(test_data.text.isnull().sum())

def remove_stopwords(column, df, stopword):
    df[column] = df[column].apply(lambda x: ' '.join((word for word in x.split() if word not in stopword)))
    return df
train_data = remove_stopwords('text', train_data, stopword_ls)
test_data = remove_stopwords('text', test_data, stopword_ls)
train_data.head()
from wordcloud import WordCloud
from matplotlib import pyplot as plt
text = ' '.join((title for title in train_data.text))
word_cloud = WordCloud(collocations=False, background_color='white').generate(text)
plt.imshow(word_cloud, interpolation='bilinear')
from wordcloud import WordCloud
from matplotlib import pyplot as plt
text_ts = ' '.join((title for title in test_data.text))
word_cloud = WordCloud(collocations=False, background_color='white').generate(text_ts)
plt.imshow(word_cloud, interpolation='bilinear')
tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')