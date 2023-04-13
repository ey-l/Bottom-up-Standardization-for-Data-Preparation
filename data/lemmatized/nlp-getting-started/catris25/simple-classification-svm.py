import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import os, re
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()

def clean_text(text):
    temp = text.lower()
    temp = re.sub('\n', ' ', temp)
    temp = re.sub("'", '', temp)
    temp = re.sub('-', ' ', temp)
    temp = re.sub('(http|https|pic.)\\S+', ' ', temp)
    temp = re.sub('[^\\w\\s]', ' ', temp)
    return temp
stop_words = ['as', 'in', 'of', 'is', 'are', 'were', 'was', 'it', 'for', 'to', 'from', 'into', 'onto', 'this', 'that', 'being', 'the', 'those', 'these', 'such', 'a', 'an']

def remove_stopwords(text):
    tokenized_words = word_tokenize(text)
    temp = [word for word in tokenized_words if word not in stop_words]
    temp = ' '.join(temp)
    return temp
_input1['clean'] = _input1['text'].apply(clean_text)
_input1['clean'] = _input1['clean'].apply(remove_stopwords)
_input1['clean']

def combine_attributes(text, keyword):
    var_list = [text, keyword]
    combined = ' '.join((x for x in var_list if x))
    return combined
_input1 = _input1.fillna('', inplace=False)
_input1['combine'] = _input1.apply(lambda x: combine_attributes(x['clean'], x['keyword']), axis=1)
X = _input1['combine']
y = _input1['target']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=99)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
clf = SVC(kernel='linear')