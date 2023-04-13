import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.sample(6)
_input1.isnull().sum()
_input1.info()
_input1.describe()
_input1.drop(['keyword', 'location'], axis=1)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
stop_word = stopwords.words('english')
print(stop_word)
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('wordnet')
nltk.download('omw-1.4')
import re

def clean(text):
    word = re.sub('[^a-zA-Z]', ' ', text)
    word = word.lower()
    word = word.split()
    lamm = [PorterStemmer().stem(s) for s in word if not s in stop_word]
    lamm = ' '.join(lamm)
    return lamm
_input1['text_cleaned'] = _input1['text'].apply(clean)
_input1
x = _input1['text_cleaned'].values
y = _input1['target'].values
print(x)
print(y)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
clf = TfidfVectorizer()
x = clf.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.1, random_state=44, stratify=y)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(penalty='l2')