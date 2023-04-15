import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
(train.shape, test.shape)
train.head()
test.head()
sns.countplot(x='target', data=train)
y = train.target
X = train.text
X
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    a_lemmas = [lemma.lower() for lemma in lemmas if lemma.isalpha() and lemma not in stopwords.words('english')]
    lemmatized_text = ' '.join(a_lemmas)
    return lemmatized_text
df_train = pd.DataFrame([])
cleaned_text = []
for text in X:
    cleaned_text.append(preprocess(text))
df_train['text'] = cleaned_text
df_train
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(df_train, y, test_size=0.3, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features=350)
X_train_bow = vect.fit_transform(X_train['text'].tolist())
X_test_bow = vect.transform(X_test['text'].tolist())
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()