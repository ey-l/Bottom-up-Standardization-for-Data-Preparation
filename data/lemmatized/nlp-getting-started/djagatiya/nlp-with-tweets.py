import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
import nltk
import string
import re
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.shape
_input1.dtypes
_input1.isnull().sum().plot(kind='bar')
plt.title('Missing values')
target_count = _input1.groupby('target').size().reset_index(name='counts')
plt.bar(target_count.target, target_count.counts)
plt.xticks([0, 1], labels=['Not disaster tweets', 'disaster tweets'])
plt.title('Target Distribution')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
eng_stopwords = set(stopwords.words('english'))

def preprocess_text(val):
    val = val.lower()
    val = re.sub('http\\S+', '', val)
    val = ''.join([c for c in val if c not in string.punctuation])
    val = re.sub('\\d', ' ', val)
    val = re.sub('\\s+', ' ', val)
    tokens = nltk.word_tokenize(val)
    tokens = [t for t in tokens if t not in eng_stopwords]
    tokens = [ps.stem(t) for t in tokens]
    return ' '.join(tokens)
_input1['clean_text'] = _input1.text.apply(preprocess_text)
_input1['clean_text_len'] = _input1.clean_text.apply(lambda x: len(x))
_input1.head()
_input1.describe()
fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(_input1['clean_text_len'][_input1.target == 0])
plt.title('Not disaster tweets')
plt.subplot(1, 2, 2)
plt.hist(_input1['clean_text_len'][_input1.target == 1], color='orange')
plt.title('Disaster tweets')
fig.supxlabel('tweet lenghts')
fig.supylabel('counts')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
(X_train, X_valid, y_train, y_valid) = train_test_split(_input1.clean_text, _input1.target, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 1))