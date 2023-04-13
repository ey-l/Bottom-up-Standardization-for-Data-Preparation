import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import re
import string
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1
_input1[_input1['target'] == 1]['text'].values[:5]
_input1[_input1['target'] == 0]['text'].values[:5]
_input1.isna().sum()
tweets = _input1[['text', 'target']]
sns.countplot(x=tweets['target'])
print(f'{tweets.target[tweets.target == 1].count() / tweets.target.count() * 100:.2f} % of tweets are labeled as disaster tweets in data')
tweets.isna().sum()
(train, test) = train_test_split(tweets, test_size=0.25, random_state=8)
stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def lower_text(text):
    """
        function to convert text into lowercase
        input: text
        output: cleaned text
    """
    text = text.lower()
    return text

def remove_newline(text):
    """
        function to remove new line characters in text
        input: text
        output: cleaned text
    """
    text = re.sub('\\n', ' ', text)
    return text

def remove_punctuations(text):
    """
        function to remove punctuations from text
        input: text
        output: cleaned text
    """
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

def remove_links(text):
    """
        function to links and urls from text
        input: text
        output: cleaned text
    """
    text = re.sub('(https|http)?:\\/\\/(\\w|\\.|\\/|\\?|\\=|\\&|\\%)*\\b', '', text, flags=re.MULTILINE)
    return text

def remove_tags(text):
    """
        function to remove references and hashtags from text
        input: text
        output: cleaned text
    """
    text = re.sub('(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|(\\w+:\\/\\/\\S+)', '', text)
    return text

def remove_multiplespaces(text):
    """
        function to remove multiple spaces from text
        input: text
        output: cleaned text
    """
    text = re.sub('\\s+', ' ', text, flags=re.I)
    return text

def remove_specialchars(text):
    """
        function to remove special characters from text
        input: text
        output: cleaned text
    """
    text = re.sub('\\W', ' ', text)
    return text

def remove_stopwords(text):
    """
        function to tokenize the words using nltk word tokenizer and remove the stop words using nltk package's english stop words
        input: text
        output: cleaned text
    """
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    return text

def word_stemming(text):
    """
        function to perform stemming using porter stemmer from nltk package
        input: text
        output: cleaned text
    """
    text = ' '.join([stemmer.stem(word) for word in word_tokenize(text)])
    return text
train.text = train.text.apply(lambda text: lower_text(text))
train.text = train.text.apply(lambda text: remove_newline(text))
train.text = train.text.apply(lambda text: remove_punctuations(text))
train.text = train.text.apply(lambda text: remove_links(text))
train.text = train.text.apply(lambda text: remove_tags(text))
train.text = train.text.apply(lambda text: remove_multiplespaces(text))
train.text = train.text.apply(lambda text: remove_specialchars(text))
train.text = train.text.apply(lambda text: remove_stopwords(text))
train.text = train.text.apply(lambda text: word_stemming(text))
test.text = test.text.apply(lambda text: lower_text(text))
test.text = test.text.apply(lambda text: remove_newline(text))
test.text = test.text.apply(lambda text: remove_punctuations(text))
test.text = test.text.apply(lambda text: remove_links(text))
test.text = test.text.apply(lambda text: remove_tags(text))
test.text = test.text.apply(lambda text: remove_multiplespaces(text))
test.text = test.text.apply(lambda text: remove_specialchars(text))
test.text = test.text.apply(lambda text: remove_stopwords(text))
test.text = test.text.apply(lambda text: word_stemming(text))
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(20, 12))
wc_disaster = WordCloud(width=800, height=600, background_color='white', stopwords=STOPWORDS).generate(' '.join(train[train.target == 1]['text']))
wc_nondisaster = WordCloud(width=800, height=600, background_color='white', stopwords=STOPWORDS).generate(' '.join(train[train.target == 0]['text']))
ax1.imshow(wc_disaster)
ax1.set_title('Word cloud of disaster tweets', fontsize=20)
ax1.axis('off')
ax2.imshow(wc_nondisaster)
ax2.set_title('Word cloud of non disaster tweets', fontsize=20)
ax2.axis('off')
fig.show()
count_vectorizer = CountVectorizer()
train_vectors_bow = count_vectorizer.fit_transform(train['text'])
test_vectors_bow = count_vectorizer.transform(test['text'])
tfidf_vectorizer = TfidfVectorizer()
train_vectors_tf = tfidf_vectorizer.fit_transform(train['text'])
test_vectors_tf = tfidf_vectorizer.transform(test['text'])
clf = LogisticRegression()
print('Bag-of-words:\n')
scores = cross_val_score(clf, train_vectors_bow, train['target'], cv=5, scoring='f1')
for (k, score) in zip(range(len(scores)), scores):
    print('F1 Score for fold %d is %.2f ' % (k + 1, score))