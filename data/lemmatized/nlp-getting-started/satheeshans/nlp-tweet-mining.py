import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0.head()
_input1.isnull().sum()
X = _input1.drop(['id'], axis=1)
X.head()
key = X['keyword'].value_counts().index[0]
print(key)
loc = X['location'].value_counts().index[0]
print(loc)
_input1['keyword'] = _input1['keyword'].fillna(_input1['keyword'].value_counts().idxmax())
_input1['location'] = _input1['location'].fillna(_input1['location'].value_counts().idxmax())
_input1.isnull().sum()
X_new = _input1.drop(['target'], axis=1)
X_new.head()
y = _input1['target']
y.shape
_input0.isnull().sum()
_input0['keyword'] = _input0['keyword'].fillna(_input0['keyword'].value_counts().idxmax())
_input0['location'] = _input0['location'].fillna(_input0['location'].value_counts().idxmax())
_input0.isnull().sum()
X_new = X_new.drop(['id'], axis=1)
X_new.head()
X_new = X_new.replace('[^a-zA-Z]', ' ', regex=True, inplace=False)
X_new.head()
for i in X_new.columns:
    X_new[i] = X_new[i].str.lower()
X_new.head()
from nltk.corpus import stopwords
stop = stopwords.words('english')
X_new['keyword'].apply(lambda x: [item for item in x if item not in stop])
X_new['location'].apply(lambda x: [item for item in x if item not in stop])
X_new['text'].apply(lambda x: [item for item in x if item not in stop])
print(X_new.shape)
ori_test = _input0.drop(['id'], axis=1)
ori_test = ori_test.replace('[^a-zA-Z]', ' ', regex=True, inplace=False)
ori_test.head()
for i in ori_test.columns:
    ori_test[i] = ori_test[i].str.lower()
ori_test['keyword'].apply(lambda x: [item for item in x if item not in stop])
ori_test['location'].apply(lambda x: [item for item in x if item not in stop])
ori_test['text'].apply(lambda x: [item for item in x if item not in stop])
print(ori_test.shape)
X_new['sentence'] = X_new['keyword'] + ' ' + X_new['text']
train_text = np.array(X_new['sentence'])
print(train_text[0])
ori_test['sentence'] = ori_test['keyword'] + ' ' + ori_test['text']
test_text = np.array(ori_test['sentence'])
print(test_text[0])
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(train_text, y, test_size=0.25, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(x_train)
keyword = vectorizer.get_feature_names()
X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)
X_test_new = vectorizer.transform(test_text)
print(x_train[0])
print('\n')
print(X_train[0])
rc = RandomForestClassifier(max_depth=400, random_state=0, n_estimators=300)