import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
import string
punct = string.punctuation

def text_data_cleaning(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
classifier = LinearSVC()
x = _input1['text']
y = _input1['target']
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=42)
clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])