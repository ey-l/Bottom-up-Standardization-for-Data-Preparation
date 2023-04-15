import numpy as np
import pandas as pd
import nltk
import re
import sys
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
print('Python {}'.format(sys.version))
print('Numpy {}'.format(np.__version__))
print('Panda {}'.format(pd.__version__))
print('NLTK {}'.format(nltk.__version__))
print('Seaborn {}'.format(sns.__version__))
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
print(df_train.info())
print('===================================')
print(df_train.head())
print('===================================')
df_train.drop(['id', 'keyword', 'location'], axis=1, inplace=True)

def avg_word(sentence):
    words = sentence.split()
    return sum((len(word) for word in words)) / len(words)
df_train['avg_word'] = df_train['text'].apply(lambda x: avg_word(x))
print('Disaster Tweets')
print('=================================')
Disaster = df_train[df_train.target == 1]
Disaster.head()
print('Non-Disaster Tweets')
print('=================================')
Non_Disaster = df_train[df_train.target == 0]
Non_Disaster.head()
classes = df_train.loc[:, 'target']
print(classes.value_counts())
df_train['text'] = df_train['text'].apply(lambda x: ' '.join((x.lower() for x in x.split())))
df_train['text'].head()
df_train['text'] = df_train['text'].str.replace('(http|ftp|https):\\/\\/[\\w\\-_]+(\\.[\\w\\-_]+)+([\\w\\-\\.,@?^=%&amp;:/~\\+#]*[\\w\\-\\@?^=%&amp;/~\\+#])?', ' ')
df_train['text'].head()
df_train['text'] = df_train['text'].str.replace('rt ', '').str.replace('@', '').str.replace('#', '').str.replace('[^\\w\\s]', '').str.replace('[1-9]', '')
df_train['text'].head()
df_train['text'] = df_train['text'].str.replace('\\d+(\\.\\d+)?', '')
df_train['text'].head()
df_train['text'] = df_train['text'].str.replace('[^\\w\\d\\s]', ' ')
df_train['text'] = df_train['text'].str.replace('^\\s+|\\s+?$', '')
df_train['text'] = df_train['text'].str.replace('\\s+', ' ')
df_train['text'].head()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
df_train['text'] = df_train['text'].apply(lambda x: ' '.join((term for term in x.split() if term not in stop_words)))
df_train['text'].head()
from nltk.stem import PorterStemmer
st = PorterStemmer()
df_train['text'] = df_train['text'].apply(lambda x: ' '.join([st.stem(word) for word in x.split()]))
df_train['text'].head()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(df_train.text).toarray()
y = df_train.iloc[:, 1].values
print(X)
print('=============================')
print(y)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
names = ['K Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'SGD Classifier', 'Naive Bayes', 'SVM Linear']
classifiers = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression(), SGDClassifier(max_iter=100), MultinomialNB(), SVC(kernel='linear')]
models = zip(names, classifiers)
for (name, model) in models:
    nltk_model = model