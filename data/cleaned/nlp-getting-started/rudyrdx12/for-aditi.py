import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
train.drop(['keyword', 'location'], axis=1, inplace=True)
train.head()
train.info()
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
X = vec.fit_transform(train['text'])
y = train['target']
x_test = vec.transform(test['text'])
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LogisticRegression(random_state=22)