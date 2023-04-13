import numpy as np
import pandas as pd
import nltk
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
pd.pandas.set_option('display.max_rows', None)
_input1
_input1['keyword'].unique()
_input1['location'].unique()
len(_input1['location'].unique())
len(_input1['keyword'].unique())
_input1.columns
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', usecols=['text', 'target'])
_input1['target'].value_counts()
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
ls = WordNetLemmatizer()
corpus = []
for i in range(0, len(_input1)):
    _input1 = re.sub('[^a-zA-Z]', ' ', _input1['text'][i])
    _input1 = _input1.lower()
    _input1 = _input1.split()
    _input1 = [ls.lemmatize(word) for word in _input1 if word not in stopwords.words('english')]
    _input1 = ' '.join(_input1)
    corpus.append(_input1)
corpus
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
X
y = _input1['target']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.naive_bayes import MultinomialNB