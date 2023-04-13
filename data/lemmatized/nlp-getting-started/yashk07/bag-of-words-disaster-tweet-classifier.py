import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head(5)
_input1.info()
sns.heatmap(_input1.isnull())
_input1 = _input1.drop(['location', 'keyword'], axis=1, inplace=False)
_input1
real = _input1[_input1['target'] == 1]
real
unreal = _input1[_input1['target'] == 0]
unreal
print('real disaster message percentage:', len(real) / len(_input1) * 100)
print('fake disaster message percentage:', len(unreal) / len(_input1) * 100)
sns.countplot(_input1['target'])
import string
string.punctuation
from nltk.corpus import stopwords
stopwords.words('english')

def message_cleaning(message):
    test_punc_removed = [char for char in message if char not in string.punctuation]
    test_punc_removed_joined = ''.join(test_punc_removed)
    test_punc_removed_joined_clean = [word for word in test_punc_removed_joined.split(' ') if word not in stopwords.words('english')]
    return test_punc_removed_joined_clean
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer=message_cleaning)
disaster_tweet_vectorizer = vectorizer.fit_transform(_input1['text'])
print(vectorizer.get_feature_names())
print(disaster_tweet_vectorizer.toarray())
disaster_tweet_vectorizer.shape
label = _input1['target']
label.shape
X = disaster_tweet_vectorizer
X = X.toarray()
X
y = label
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
LR = LogisticRegression()
DTC = DecisionTreeClassifier()
RFC = RandomForestClassifier()
NB = GaussianNB()