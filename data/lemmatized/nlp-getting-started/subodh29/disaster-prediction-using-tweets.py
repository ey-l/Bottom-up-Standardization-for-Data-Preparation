import numpy as np
import pandas as pd
import string
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
missing_values = ['na', 'n/a', '-', 'NaN']
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', na_values=missing_values)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv', na_values=missing_values)
_input1.head()

def text_process(tweet):
    tweet = re.sub('http\\S+', '', tweet)
    stopword = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    word_tokens = word_tokenize(tweet)
    word_tokens_temp = []
    for word in word_tokens:
        word = ''.join((i for i in word if not i.isdigit()))
        word_tokens_temp.append(word)
    filtered_words = [lem.lemmatize(w) for w in word_tokens_temp if w not in stopword and w not in string.punctuation]
    new_sentence = ' '.join(filtered_words)
    return new_sentence
_input1['text'] = _input1['text'].apply(text_process)
stopwordSet = set(STOPWORDS)
tweet_words = ''
for tweet in _input1['text']:
    tweet = str(tweet)
    tokens = tweet.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        tweet_words += ' '.join(tokens) + ' '
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwordSet, min_font_size=10).generate(tweet_words)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
_input0.head()
_input0['text'] = _input0['text'].apply(text_process)
y = _input1['target']
x = _input1['text'] + ' ' + _input1['keyword']
(xtrain, xvalid, ytrain, yvalid) = train_test_split(x.values.astype(str), y, stratify=y, random_state=42, test_size=0.1, shuffle=True)
print(xtrain.shape)
print(xvalid.shape)
tfv = TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', token_pattern='\\w{1,}', ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')