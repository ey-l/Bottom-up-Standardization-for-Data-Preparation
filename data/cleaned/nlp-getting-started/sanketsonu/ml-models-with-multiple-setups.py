import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
train = train.drop(['keyword', 'location'], axis=1)
test = test.drop(['keyword', 'location'], axis=1)
train.head()
print('Shape of Train set:', train.shape)
print('Shape of Test set:', test.shape)
train = train.drop_duplicates(subset=['text'], keep='last')
print('Shape of Train set after removing duplicates:', train.shape)
train[train['text'].map(lambda x: x.isascii())]
test[test['text'].map(lambda x: x.isascii())]

def clean_tweets(text):
    text = re.sub('@[A-Za-z0-9_]+', '', text)
    text = re.sub('#', '', text)
    text = re.sub('RT[\\s]+', ' ', text)
    text = re.sub('\\n', '', text)
    text = re.sub(',', '', text)
    text = re.sub('.[.]+', '', text)
    text = re.sub('\\w+:\\/\\/\\S+', '', text)
    text = re.sub('https?:\\/\\/\\S+', '', text)
    text = re.sub('/', ' ', text)
    text = re.sub('-', ' ', text)
    text = re.sub('_', ' ', text)
    text = re.sub('!', '', text)
    text = re.sub(':', ' ', text)
    text = re.sub('$', '', text)
    text = re.sub('%', '', text)
    text = re.sub('^', '', text)
    text = re.sub('&', '', text)
    text = re.sub('=', ' ', text)
    text = re.sub(' +', ' ', text)
    return text

def clean_emoji(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
train['text'] = train['text'].apply(clean_tweets)
train['text'] = train['text'].apply(clean_emoji)
train['text'] = train.text.str.lower()
train['text'] = train['text'].str.strip()
test['text'] = test['text'].apply(clean_tweets)
test['text'] = test['text'].apply(clean_emoji)
test['text'] = test.text.str.lower()
test['text'] = test['text'].str.strip()
pd.set_option('display.max_colwidth', -1)
train['target'].value_counts()
model_params = {'SVC': {'model': SVC(), 'params': {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}}, 'MultinomialNB': {'model': MultinomialNB(), 'params': {'alpha': np.linspace(0.5, 1.5, 6), 'fit_prior': [True, False]}}, 'logistics_regression': {'model': LogisticRegression(solver='lbfgs', multi_class='auto'), 'params': {'C': [0.1, 1, 20, 40, 60, 80, 100], 'solver': ['lbfgs', 'liblinear']}}, 'random_forest': {'model': RandomForestClassifier(), 'params': {'n_estimators': [80, 85, 90, 95, 100], 'max_depth': [20, 30, None], 'criterion': ['gini', 'entropy']}}}
df = train.copy()
import string
string.punctuation
punctuations_list = string.punctuation

def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
df['text'] = df['text'].apply(lambda x: cleaning_punctuations(x))
X = df['text']
y = df['target']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=3)
vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)