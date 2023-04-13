import numpy as np
import pandas as pd
import string
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input2
_input1.shape
_input1.text.isnull().values.any()
_input1.location.value_counts()
_input1.keyword.value_counts()
_input1.text.describe()
_input1.keyword.describe()
_input1.location.describe()
_input1 = _input1.drop(['location', 'keyword'], axis=1, inplace=False)
_input1
_input1.text[:3]
_input1.columns
_input1['text'] = _input1['text'].str.replace('.', '')
_input1['text'] = _input1['text'].str.replace(',', '')
_input1['text'] = _input1['text'].str.replace('&', '')
_input1['text'] = _input1['text'].str.lower()
string.punctuation

def remove_punctuation(text):
    without_punct = ''.join([i for i in text if i not in string.punctuation])
    return without_punct
_input1['text'] = _input1['text'].apply(lambda x: remove_punctuation(x))

def tokenize(string):
    """
    Tokenizes the string to a list of words
    """
    word_tokens = string.split()
    return word_tokens
_input1['text'] = _input1['text'].apply(lambda x: tokenize(x))
_input1.head()
stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    output = [i for i in text if i not in stopwords]
    return output
_input1['text'] = _input1['text'].apply(lambda x: remove_stopwords(x))
porter_stemmer = PorterStemmer()

def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
_input1['text'] = _input1['text'].apply(lambda x: stemming(x))
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
nltk.download('wordnet')
_input1['text'] = _input1['text'].apply(lambda x: lemmatizer(x))
_input1['text_strings'] = _input1['text'].apply(lambda x: ' '.join([str(word) for word in x]))
_input1['text_strings'].head()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(_input1['text_strings'])
x_train = X.toarray()
x_train = np.array(x_train)
x_train.shape
y_train = _input1['target']
y_train.shape
clf = LogisticRegression(random_state=42)