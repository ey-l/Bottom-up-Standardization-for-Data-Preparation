import pandas as pd
import numpy as np
import re
import nltk
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.shape
text = list(_input1['text'])
list1 = []
for doc in text:
    doc = doc.lower()
    list1.append(doc)
text = list1
list1 = []
for doc in text:
    doc = word_tokenize(doc)
    list1.append(doc)
text = list1
punc = '!()-[]{};:\'"\\, <>./?@#$%^&*_~'
list2 = []
for doc in text:
    list1 = []
    for words in doc:
        for char in words:
            if char in punc:
                words = words.replace(char, '')
        if len(words) > 0:
            list1.append(words)
    list2.append(list1)
text = list2
list2 = []
for doc in text:
    list1 = []
    for words in doc:
        if words not in stopwords.words('english'):
            list1.append(words)
    list2.append(list1)
text = list2
wordnet = WordNetLemmatizer()
text_lemmatize = []
for doc in text:
    list1 = []
    for word in doc:
        list1.append(wordnet.lemmatize(word))
    text_lemmatize.append(list1)
text = text_lemmatize
list1 = []
for doc in text:
    string = ' '.join(doc)
    list1.append(string)
text = list1
Y = list(_input1['target'])
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(text, Y, test_size=0.0001, random_state=0)
print(len(X_train), ' ', len(X_test))
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', decode_error='ignore')