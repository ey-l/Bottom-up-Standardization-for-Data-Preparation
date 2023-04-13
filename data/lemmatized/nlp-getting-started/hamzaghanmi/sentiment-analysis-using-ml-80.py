import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
import string
from sklearn import preprocessing
from sklearn.manifold import TSNE
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from time import time
from sklearn.model_selection import StratifiedKFold
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
print(_input1.shape, _input0.shape)
_input1.head(3)
_input1.isnull().sum()
_input1 = _input1.drop_duplicates(subset=['text', 'target'], keep='first')
_input1.shape
_input1.target.value_counts()
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))
_input1.groupby('target').id.count().plot.bar(ylim=0)
_input1['text_length'] = _input1.text.apply(lambda x: len(x.split()))
_input0['text_length'] = _input0.text.apply(lambda x: len(x.split()))
_input1['text_length'].describe()
_input0['text_length'].describe()

def plot_word_count(df, data_name):
    sns.distplot(df['text_length'].values)
    plt.title(f'Sequence char count: {data_name}')
    plt.grid(True)
plt.subplot(1, 2, 1)
plot_word_count(_input1, 'Train')
plt.subplot(1, 2, 2)
plot_word_count(_input0, 'Test')
plt.subplots_adjust(right=3.0)
list_ = []
for i in _input1.text:
    list_ += i
list_ = ''.join(list_)
allWords = list_.split()
vocabulary = set(allWords)
len(vocabulary)

def create_corpus(df, target):
    corpus = []
    for x in df[df['target'] == target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus
import collections
allWords = create_corpus(_input1, target=0)
vocabulary = set(allWords)
vocabulary_list = list(vocabulary)
plt.figure(figsize=(16, 5))
counter = collections.Counter(allWords)
most = counter.most_common()
x = []
y = []
for (word, count) in most[:20]:
    x.append(word)
    y.append(count)
sns.barplot(x=y, y=x)
import collections
allWords = create_corpus(_input1, target=1)
vocabulary = set(allWords)
vocabulary_list = list(vocabulary)
plt.figure(figsize=(16, 5))
counter = collections.Counter(allWords)
most = counter.most_common()
x = []
y = []
for (word, count) in most[:20]:
    x.append(word)
    y.append(count)
sns.barplot(x=y, y=x)
string.punctuation
text = 'hey # how are !you doing ?'
''.join([char for char in text if char not in string.punctuation])
text = 'hey 4 look 333 at me0'
re.sub('[0-9]', '', text)
stopwords.words('english')
text = 'hey this is me and I am here to help you  '
tokens = word_tokenize(text)
tokens = [word for word in tokens if word not in stopwords.words('english')]
' '.join(tokens)
pstem = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub('[0-9]', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [pstem.stem(word) for word in tokens]
    text = ' '.join(tokens)
    return text
clean_text("hey I am here # ! looks 4 GOOD can't see you!")
_input1['clean'] = _input1['text'].apply(clean_text)
_input0['clean'] = _input0['text'].apply(clean_text)
_input1[['text', 'clean']].head(4)
list_ = []
for i in _input1.clean:
    list_ += i
list_ = ''.join(list_)
allWords = list_.split()
vocabulary = set(allWords)
len(vocabulary)
tfidf = TfidfVectorizer(sublinear_tf=True, max_features=60000, min_df=1, norm='l2', ngram_range=(1, 2))
features = tfidf.fit_transform(_input1.clean).toarray()
features.shape
features_test = tfidf.transform(_input0.clean).toarray()
skf = StratifiedKFold(n_splits=4, random_state=48, shuffle=True)
accuracy = []
n = 1
y = _input1['target']
for (trn_idx, test_idx) in skf.split(features, y):
    start_time = time()
    (X_tr, X_val) = (features[trn_idx], features[test_idx])
    (y_tr, y_val) = (y.iloc[trn_idx], y.iloc[test_idx])
    model = LogisticRegression(max_iter=1000, C=3)