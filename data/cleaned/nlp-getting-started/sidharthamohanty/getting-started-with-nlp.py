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
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
df_train
df_test = pd.read_csv('data/input/nlp-getting-started/test.csv')
df_test
sample_submission = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
sample_submission
df_train.shape
df_train.text.isnull().values.any()
df_train.location.value_counts()
df_train.keyword.value_counts()
df_train.text.describe()
df_train.keyword.describe()
df_train.location.describe()
df_train.drop(['location', 'keyword'], axis=1, inplace=True)
df_train
df_train.text[:3]
df_train.columns
df_train['text'] = df_train['text'].str.replace('.', '')
df_train['text'] = df_train['text'].str.replace(',', '')
df_train['text'] = df_train['text'].str.replace('&', '')
df_train['text'] = df_train['text'].str.lower()
string.punctuation

def remove_punctuation(text):
    without_punct = ''.join([i for i in text if i not in string.punctuation])
    return without_punct
df_train['text'] = df_train['text'].apply(lambda x: remove_punctuation(x))

def tokenize(string):
    """
    Tokenizes the string to a list of words
    """
    word_tokens = string.split()
    return word_tokens
df_train['text'] = df_train['text'].apply(lambda x: tokenize(x))
df_train.head()
stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    output = [i for i in text if i not in stopwords]
    return output
df_train['text'] = df_train['text'].apply(lambda x: remove_stopwords(x))
porter_stemmer = PorterStemmer()

def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
df_train['text'] = df_train['text'].apply(lambda x: stemming(x))
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
nltk.download('wordnet')
df_train['text'] = df_train['text'].apply(lambda x: lemmatizer(x))
df_train['text_strings'] = df_train['text'].apply(lambda x: ' '.join([str(word) for word in x]))
df_train['text_strings'].head()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_train['text_strings'])
x_train = X.toarray()
x_train = np.array(x_train)
x_train.shape
y_train = df_train['target']
y_train.shape
clf = LogisticRegression(random_state=42)