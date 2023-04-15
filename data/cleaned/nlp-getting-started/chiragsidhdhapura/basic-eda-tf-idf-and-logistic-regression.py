import numpy as np
import pandas as pd
import nltk
train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
(train_data.shape, test_data.shape)
train_data.head()
test_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.keyword.unique()
train_text = train_data.text
test_text = test_data.text
y = train_data.target
import re

def clean_text(text):
    text = text.lower()
    text = re.sub('#', '', text)
    text = re.sub('[^a-zA-Z ]', '', text)
    return text
train_text.head()
train_text = train_text.apply(clean_text)
test_text = test_text.apply(clean_text)
train_text.head()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
train_sequence = []
for i in range(len(train_text)):
    words = nltk.word_tokenize(train_text.iloc[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sent = ' '.join(words)
    train_sequence.append(sent)
len(train_sequence)
test_sequence = []
for i in range(len(test_text)):
    words = nltk.word_tokenize(test_text.iloc[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sent = ' '.join(words)
    test_sequence.append(sent)
len(test_sequence)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 3), max_features=10000)
vectorized_train = tfidf.fit_transform(train_sequence)
vectorized_train.shape
vectorized_test = tfidf.transform(test_sequence)
vectorized_test.shape
vectorized_train = vectorized_train.toarray()
vectorized_test = vectorized_test.toarray()
vectorized_train[0]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(vectorized_train, y, test_size=0.2, random_state=0)
classifier = LogisticRegression(C=3)