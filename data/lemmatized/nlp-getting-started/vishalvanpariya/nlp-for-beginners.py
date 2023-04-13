import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1 = _input1.drop('keyword', 1)
_input1 = _input1.drop('location', 1)
_input0 = _input0.drop('keyword', 1)
_input0 = _input0.drop('location', 1)
y_train = _input1.iloc[:, -1]
x_train = _input1.iloc[:, :-1]
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def atcontain(text):
    ar = []
    text = text.split()
    for t in text:
        if '@' in t:
            ar.append('TAGSOMEBODY')
        else:
            ar.append(t)
    return ' '.join(ar)

def dataclean(data):
    corpus = []
    for i in range(data.shape[0]):
        tweet = data.iloc[i, -1]
        tweet = atcontain(tweet)
        tweet = re.sub('http\\S+', '', tweet)
        tweet = re.sub('[^a-zA-z]', ' ', tweet)
        tweet = tweet.lower()
        tweet = word_tokenize(tweet)
        tweet = [lemmatizer.lemmatize(word) for word in tweet if word not in stopwords.words('english')]
        tweet = [word for word in tweet if word not in set(string.punctuation)]
        tweet = ' '.join(tweet)
        corpus.append(tweet)
    return corpus
x_corpus_train = dataclean(x_train)
x_corpus_test = dataclean(_input0)
dic = defaultdict(int)
for text in x_corpus_train:
    text = text.split()
    for word in text:
        dic[word] = dic[word] + 1
sorted_data = sorted(dic.items(), key=lambda x: x[1], reverse=True)
sorted_data[:20]
cv = TfidfVectorizer(max_features=8000)
x_train_vector = cv.fit_transform(x_corpus_train).toarray()
x_test_vector = cv.transform(x_corpus_test).toarray()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()