import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re, string, unicodedata
from pandas import DataFrame
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
print(len(_input1))
_input1.head()
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
print(len(_input0))
_input0.head()
cols = _input1.columns[:30]
colours = ['#000099', '#ffff00']
sns.heatmap(_input1[cols].isnull(), cmap=sns.color_palette(colours))
cols = _input0.columns[:30]
colours = ['#000099', '#ffff00']
sns.heatmap(_input1[cols].isnull(), cmap=sns.color_palette(colours))
_input1 = _input1.drop('keyword', axis=1, inplace=False)
_input0 = _input0.drop('keyword', axis=1, inplace=False)
_input1 = _input1.drop('location', axis=1, inplace=False)
_input0 = _input0.drop('location', axis=1, inplace=False)

def remove_URL(sample):
    return re.sub('http\\S+', '', sample)

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub('[^\\w\\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def preprocess(sample):
    sample = remove_URL(sample)
    words = nltk.word_tokenize(sample)
    return normalize(words)
vocabulary = []
new_train = []
for text in _input1['text']:
    new_text = preprocess(text)
    vocabulary.append(new_text)
    new_train.append(' '.join(new_text))
new_test = []
for text in _input0['text']:
    new_text = preprocess(text)
    new_test.append(' '.join(new_text))
tokens = [item for sublist in vocabulary for item in sublist]
print(len(tokens))
frequency_dist = nltk.FreqDist(tokens)
sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:50]
wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud)
plt.axis('off')
final_train = DataFrame(new_train, columns=['text'])
final_train['id'] = _input1['id']
final_train['target'] = _input1['target']
final_train.head()
final_test = DataFrame(new_test, columns=['text'])
final_test['id'] = _input0['id']
final_test.head()
X_train = final_train.loc[:7613, 'text'].values
y_train = final_train.loc[:7613, 'target'].values
X_test = final_test.loc[:3263, 'text'].values
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)
from sklearn.naive_bayes import MultinomialNB