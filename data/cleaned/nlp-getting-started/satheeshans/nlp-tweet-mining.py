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
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
train_df.head()
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
test_df.head()
train_df.isnull().sum()
X = train_df.drop(['id'], axis=1)
X.head()
key = X['keyword'].value_counts().index[0]
print(key)
loc = X['location'].value_counts().index[0]
print(loc)
train_df['keyword'] = train_df['keyword'].fillna(train_df['keyword'].value_counts().idxmax())
train_df['location'] = train_df['location'].fillna(train_df['location'].value_counts().idxmax())
train_df.isnull().sum()
X_new = train_df.drop(['target'], axis=1)
X_new.head()
y = train_df['target']
y.shape
test_df.isnull().sum()
test_df['keyword'] = test_df['keyword'].fillna(test_df['keyword'].value_counts().idxmax())
test_df['location'] = test_df['location'].fillna(test_df['location'].value_counts().idxmax())
test_df.isnull().sum()
X_new = X_new.drop(['id'], axis=1)
X_new.head()
X_new.replace('[^a-zA-Z]', ' ', regex=True, inplace=True)
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
ori_test = test_df.drop(['id'], axis=1)
ori_test.replace('[^a-zA-Z]', ' ', regex=True, inplace=True)
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