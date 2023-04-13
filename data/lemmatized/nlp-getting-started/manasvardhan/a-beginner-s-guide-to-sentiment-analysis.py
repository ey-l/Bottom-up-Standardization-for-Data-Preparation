import pandas as pd
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1['target'].value_counts()
_input1.isna().sum()
_input1 = _input1.fillna('', inplace=False)
import string
string.punctuation

def remove_punctuation(text):
    punctuationfree = ''.join([i for i in text if i not in string.punctuation])
    return punctuationfree
_input1['text'] = _input1['text'].apply(lambda x: remove_punctuation(x))
_input1['text'] = _input1['text'].apply(lambda x: x.lower())

def tokenize(string):
    """
    Tokenizes the string to a list of words
    """
    tokens = string.split()
    return tokens
_input1['text'] = _input1['text'].apply(lambda x: tokenize(x))
_input1.head()
_input1['keyword'] = _input1['keyword'].apply(lambda x: tokenize(x))
_input1.tail()
_input1 = _input1.drop(columns=['id'], inplace=False)
_input1.head()
import nltk
stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    output = [i for i in text if i not in stopwords]
    return output
_input1['text'] = _input1['text'].apply(lambda x: remove_stopwords(x))
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
_input1['text'] = _input1['text'].apply(lambda x: stemming(x))
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
nltk.download('wordnet')
vocab = []
"\nWe add all the lists of tokenized strings to make one large list of words\n\nNote ['a','b'] + ['c'] = ['a','b','c']\n\n"
for i in _input1['text'].values:
    vocab = vocab + i
print(len(vocab))
set_vocab = set(vocab)
vocab = list(set_vocab)
print(len(vocab), type(vocab))
_input1['text_strings'] = _input1['text'].apply(lambda x: ' '.join([str(elem) for elem in x]))
_input1['text_strings'].head()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(_input1['text_strings'])
x_train = X.toarray()
import numpy as np
import numpy as nper
x_train = np.array(x_train)
y_train = _input1['target']
x_train.shape
y_train.shape
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42)