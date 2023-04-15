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
train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_data.head()
train_data_description = train_data.describe()
print(train_data_description)
train_data_info = train_data.info()
print(train_data_info)
train_data = train_data.drop(columns=['keyword', 'location'])
train_data.head()
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
train_data['text_cleaned'] = train_data['text'].apply(clean)
x = train_data['text_cleaned'].values
y = train_data['target'].values
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
classifier = TfidfVectorizer()
x = classifier.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.1, random_state=44, stratify=y)
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(penalty='l2')