import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.info()
_input1.head()
_input1['msg_len'] = _input1['text'].apply(len)
_input1['msg_len']
_input1['msg_len'].describe()
_input1[_input1['msg_len'] == 157]['text'].iloc[0]
sns.barplot(x=_input1['target'], y=_input1['msg_len'], data=_input1)
sns.set_style('darkgrid')
_input1.hist(column='msg_len', by='target', sharey=True, bins=50, figsize=(12, 4))
_input1 = _input1.drop(['id', 'keyword', 'location', 'msg_len'], axis=1, inplace=False)
import string
import time
from nltk.corpus import stopwords
commonwords = stopwords.words('english')
from nltk.corpus import words
import re
from nltk.stem.porter import PorterStemmer
string.punctuation
corpus = []
for i in range(0, _input1.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', _input1['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = ' '.join(review)
    corpus.append(review)
time.sleep(3)
corpus[0:5]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = _input1.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
nb_model = MultinomialNB()