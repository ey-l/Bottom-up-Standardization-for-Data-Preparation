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
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1 = _input1.drop(['keyword', 'location'], axis=1)
_input0 = _input0.drop(['keyword', 'location'], axis=1)
_input1.head()
print('Shape of Train set:', _input1.shape)
print('Shape of Test set:', _input0.shape)
_input1 = _input1.drop_duplicates(subset=['text'], keep='last')
print('Shape of Train set after removing duplicates:', _input1.shape)
_input1[_input1['text'].map(lambda x: x.isascii())]
_input0[_input0['text'].map(lambda x: x.isascii())]

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
_input1['text'] = _input1['text'].apply(clean_tweets)
_input1['text'] = _input1['text'].apply(clean_emoji)
_input1['text'] = _input1.text.str.lower()
_input1['text'] = _input1['text'].str.strip()
_input0['text'] = _input0['text'].apply(clean_tweets)
_input0['text'] = _input0['text'].apply(clean_emoji)
_input0['text'] = _input0.text.str.lower()
_input0['text'] = _input0['text'].str.strip()
pd.set_option('display.max_colwidth', -1)
_input1['target'].value_counts()
model_params = {'SVC': {'model': SVC(), 'params': {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}}, 'MultinomialNB': {'model': MultinomialNB(), 'params': {'alpha': np.linspace(0.5, 1.5, 6), 'fit_prior': [True, False]}}, 'logistics_regression': {'model': LogisticRegression(solver='lbfgs', multi_class='auto'), 'params': {'C': [0.1, 1, 20, 40, 60, 80, 100], 'solver': ['lbfgs', 'liblinear']}}, 'random_forest': {'model': RandomForestClassifier(), 'params': {'n_estimators': [80, 85, 90, 95, 100], 'max_depth': [20, 30, None], 'criterion': ['gini', 'entropy']}}}
df = _input1.copy()
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