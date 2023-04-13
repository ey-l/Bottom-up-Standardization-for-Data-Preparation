import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
print('Important libraries loaded successfully')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
print('Data shape = ', _input1.shape)
_input1.head()
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(_input1.shape[1])
_input1 = _input1.drop(['location', 'keyword'], axis=1)
print('location and keyword columns droped successfully')
_input1 = _input1.drop('id', axis=1)
print('id column droped successfully')
_input1.columns
_input1['text'].head(10)
corpus = []
pstem = PorterStemmer()
for i in range(_input1['text'].shape[0]):
    tweet = re.sub('[^a-zA-Z]', ' ', _input1['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
print('Corpus created successfully')
print(pd.DataFrame(corpus)[0].head(10))
rawTexData = _input1['text'].head(10)
cleanTexData = pd.DataFrame(corpus, columns=['text after cleaning']).head(10)
frames = [rawTexData, cleanTexData]
result = pd.concat(frames, axis=1, sort=False)
result
uniqueWordFrequents = {}
for tweet in corpus:
    for word in tweet.split():
        if word in uniqueWordFrequents.keys():
            uniqueWordFrequents[word] += 1
        else:
            uniqueWordFrequents[word] = 1
uniqueWordFrequents = pd.DataFrame.from_dict(uniqueWordFrequents, orient='index', columns=['Word Frequent'])
uniqueWordFrequents = uniqueWordFrequents.sort_values(by=['Word Frequent'], inplace=False, ascending=False)
uniqueWordFrequents.head(10)
uniqueWordFrequents['Word Frequent'].unique()
uniqueWordFrequents = uniqueWordFrequents[uniqueWordFrequents['Word Frequent'] >= 20]
print(uniqueWordFrequents.shape)
uniqueWordFrequents
counVec = CountVectorizer(max_features=uniqueWordFrequents.shape[0])
bagOfWords = counVec.fit_transform(corpus).toarray()
X = bagOfWords
y = _input1['target']
print('X shape = ', X.shape)
print('y shape = ', y.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=55, shuffle=True)
print('data splitting successfully')
decisionTreeModel = DecisionTreeClassifier(criterion='entropy', max_depth=None, splitter='best', random_state=55)