import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
train_data_description = _input1.describe()
print(train_data_description)
train_data_info = _input1.info()
print(train_data_info)
_input1 = _input1.drop(columns=['keyword', 'location'])
_input1.head()
import re
import nltk
from nltk.stem import PorterStemmer

def clean(text):
    pattern = re.compile('[^a-zA-Z]')
    words = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [PorterStemmer().stem(word) for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text
_input1['text_cleaned'] = _input1['text'].apply(clean)
x = _input1['text_cleaned'].values
y = _input1['target'].values
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
classifier = TfidfVectorizer()
x = classifier.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.1, random_state=44, stratify=y)
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(penalty='l2')