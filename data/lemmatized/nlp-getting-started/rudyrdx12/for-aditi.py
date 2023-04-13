import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1 = _input1.drop(['keyword', 'location'], axis=1, inplace=False)
_input1.head()
_input1.info()
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
english_stopwords = stopwords.words('english')
english_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
vec = TfidfVectorizer()
X = vec.fit_transform(_input1['text'])
y = _input1['target']
x_test = vec.transform(_input0['text'])
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LogisticRegression(random_state=22)