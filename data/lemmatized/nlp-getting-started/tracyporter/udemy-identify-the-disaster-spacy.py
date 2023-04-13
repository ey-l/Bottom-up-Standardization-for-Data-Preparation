import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1
_input0
_input2
target = _input1.target
_input1 = _input1.drop('target', axis=1, inplace=False)
_input1
combi = _input1.append(_input0)
combi
length_train = _input1['text'].str.len()
length_test = _input0['text'].str.len()
plt.hist(length_train, bins=20, label='train_tweets')
plt.hist(length_test, bins=20, label='test_tweets')
plt.legend()
target.value_counts()
percentage_disaster = target.value_counts() / len(_input1) * 100
percentage_disaster
sns.distplot(target)
plt.boxplot(target)
tweets = combi['text']
count_words = tweets.str.findall('(\\w+)').str.len()
print(count_words.sum())
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
' Cleaning Tweets '
tweets = tweets.str.lower()
tweets = tweets.apply(lambda x: ' '.join([lemmatizer.lemmatize(i) for i in re.sub('[^a-zA-Z]', ' ', x).split() if i not in stopwords]).lower())
tweets = tweets.apply(lambda x: re.sub('[^a-z]\\s', '', x))
tweets = tweets.str.replace('#', '')
tweets = tweets.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2 and len(w) < 8]))
tweets = tweets.apply(lambda x: ' '.join((word for word in x.split() if word not in stopwords)))
count_words = tweets.str.findall('(\\w+)').str.len()
print(count_words.sum())
most_freq_words = pd.Series(' '.join(tweets).lower().split()).value_counts()[:1]
tweets = tweets.apply(lambda x: ' '.join((word for word in x.split() if word not in most_freq_words)))
print(most_freq_words)
count_words = tweets.str.findall('(\\w+)').str.len()
print(count_words.sum())
from collections import Counter
from itertools import chain
v = tweets.str.split().tolist()
c = Counter(chain.from_iterable(v))
tweets = [' '.join([j for j in i if c[j] > 1]) for i in v]
total_word = 0
for (x, word) in enumerate(tweets):
    num_word = len(word.split())
    total_word = total_word + num_word
print(total_word)
import spacy
import spacy.cli
spacy.cli.download('en_vectors_web_lg')
nlp = spacy.load('en_vectors_web_lg')
import spacy
import en_vectors_web_lg
nlp = en_vectors_web_lg.load()
document = nlp(tweets[0])
print('Document : ', document)
print('Tokens : ')
for token in document:
    print(token.text)
document = nlp(tweets[0])
print(document)
for token in document:
    print(token.text, token.vector.shape)
document = nlp.pipe(tweets)
tweets_vector = np.array([tweet.vector for tweet in document])
print(tweets_vector.shape)
y = target
X = tweets_vector[:len(_input1)]
X_test = tweets_vector[len(_input1):]
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42, shuffle=True)
(X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape)
from sklearn.neural_network import MLPClassifier