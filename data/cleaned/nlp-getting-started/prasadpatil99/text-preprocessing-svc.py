import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
train = train[['text', 'target']]
test = test[['id', 'text']]
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
import re
train['text'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
test['text'] = test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
corpus_train = train['text']
corpus_test = test['text']

def replace(text):
    text = text.str.replace('^.+@[^\\.].*\\.[a-z]{2,}$', ' ')
    text = text.str.replace('\\W+', ' ')
    text = text.str.replace(' ', ' ')
    text = text.str.replace('\\d+', ' ')
    text = text.str.lower()
    return text
corpus_train = replace(corpus_train)
corpus_test = replace(corpus_test)
import nltk
nltk.download('wordnet')
from textblob import Word
freq = pd.Series(' '.join(corpus_train).split()).value_counts()[-19500:]
corpus_train = corpus_train.apply(lambda x: ' '.join((x for x in x.split() if x not in freq)))
freq.head()
freq = pd.Series(' '.join(corpus_test).split()).value_counts()[-10000:]
corpus_test = corpus_test.apply(lambda x: ' '.join((x for x in x.split() if x not in freq)))
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def wordcloud(text):
    wordcloud = WordCloud(background_color='white', max_words=500, max_font_size=30, scale=3, random_state=5).generate(str(corpus_train))
    fig = plt.figure(figsize=(15, 12))
    plt.axis('off')
    plt.imshow(wordcloud)

wordcloud(corpus_train)
import seaborn as sns
target = train['target']
sns.countplot(target)
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf_vect = TfidfVectorizer(max_features=7000)