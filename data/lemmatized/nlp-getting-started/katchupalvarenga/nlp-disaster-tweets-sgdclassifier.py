import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
(_input1.shape, _input0.shape)
data = pd.concat([_input1, _input0], axis=0)
data.head()
data.isnull().sum()
data['text'] = data['text'].str.lower()
data['keyword'] = data['keyword'].str.lower()
data['location'] = data['location'].str.lower()
data = data.set_index('id', inplace=False)
data['text'] = data['text'].apply(lambda x: re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \\t])|(\\w+:\\/\\/\\S+)|^rt|http.+?', '', x))

def remove_words(data, col):
    stop = stopwords.words('english')
    list_of_lists = data[col].str.split()
    for (idx, _) in data.iterrows():
        data[col].at[idx] = [word for word in list_of_lists[idx] if word not in stop]
remove_words(data, 'text')

def get_word_variation(data, col):
    lemmatizer = WordNetLemmatizer()
    for (idx, _) in data.iterrows():
        data[col].at[idx] = [lemmatizer.lemmatize(palavra, 'v') for palavra in data[col][idx]]
get_word_variation(data, 'text')
data.loc[data['keyword'].notnull()].head(10)
uniq_keyword = list(data['keyword'].unique())
uniq_location = list(data['location'].unique())
for i in range(len(data)):
    if data['keyword'].isnull()[i]:
        for n in data['text'][i]:
            if n in uniq_keyword:
                data['keyword'][i] = n
data.isnull().sum()
for i in range(len(data)):
    if data['location'].isnull()[i]:
        for n in data['text'][i]:
            if n in uniq_location:
                data['location'][i] = n
data.isnull().sum()
data['location'].unique()
data
data['text'] = data['text'].apply(lambda x: ' '.join(x))
data.isnull().sum()
data.loc[data['keyword'].isnull()]
data.columns
data.head()
data['keyword'] = data['keyword'].fillna('None')
data['location'] = data['location'].fillna('None')
data.head()
_input1 = data.loc[data['target'].notnull()]
_input1
_input0 = data.loc[data['target'].isnull()]
_input0 = _input0.drop('target', axis=1, inplace=False)
_input0
(_input1.shape, _input0.shape)
X = _input1['text']
y = _input1['target']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=13)
(X_train.shape, y_train.shape)
X_train
y_train
sgd = Pipeline([('countVector', CountVectorizer()), ('tfidf', TfidfTransformer()), ('modelo', SGDClassifier())])