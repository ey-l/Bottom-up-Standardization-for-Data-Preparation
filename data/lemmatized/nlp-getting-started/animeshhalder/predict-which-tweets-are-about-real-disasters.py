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
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
print(_input1.info())
print('===================================')
print(_input1.head())
print('===================================')
_input1 = _input1.drop(['id', 'keyword', 'location'], axis=1, inplace=False)

def avg_word(sentence):
    words = sentence.split()
    return sum((len(word) for word in words)) / len(words)
_input1['avg_word'] = _input1['text'].apply(lambda x: avg_word(x))
print('Disaster Tweets')
print('=================================')
Disaster = _input1[_input1.target == 1]
Disaster.head()
print('Non-Disaster Tweets')
print('=================================')
Non_Disaster = _input1[_input1.target == 0]
Non_Disaster.head()
classes = _input1.loc[:, 'target']
print(classes.value_counts())
_input1['text'] = _input1['text'].apply(lambda x: ' '.join((x.lower() for x in x.split())))
_input1['text'].head()
_input1['text'] = _input1['text'].str.replace('(http|ftp|https):\\/\\/[\\w\\-_]+(\\.[\\w\\-_]+)+([\\w\\-\\.,@?^=%&amp;:/~\\+#]*[\\w\\-\\@?^=%&amp;/~\\+#])?', ' ')
_input1['text'].head()
_input1['text'] = _input1['text'].str.replace('rt ', '').str.replace('@', '').str.replace('#', '').str.replace('[^\\w\\s]', '').str.replace('[1-9]', '')
_input1['text'].head()
_input1['text'] = _input1['text'].str.replace('\\d+(\\.\\d+)?', '')
_input1['text'].head()
_input1['text'] = _input1['text'].str.replace('[^\\w\\d\\s]', ' ')
_input1['text'] = _input1['text'].str.replace('^\\s+|\\s+?$', '')
_input1['text'] = _input1['text'].str.replace('\\s+', ' ')
_input1['text'].head()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
_input1['text'] = _input1['text'].apply(lambda x: ' '.join((term for term in x.split() if term not in stop_words)))
_input1['text'].head()
from nltk.stem import PorterStemmer
st = PorterStemmer()
_input1['text'] = _input1['text'].apply(lambda x: ' '.join([st.stem(word) for word in x.split()]))
_input1['text'].head()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(_input1.text).toarray()
y = _input1.iloc[:, 1].values
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