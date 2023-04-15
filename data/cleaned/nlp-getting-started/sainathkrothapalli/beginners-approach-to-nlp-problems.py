import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
train.head()
train.shape
train.info()
train['target'].value_counts()
import seaborn as sns
sns.countplot(train['target'])
train['target'].value_counts() / train.shape[0] * 100
train.isnull().sum()
train['keyword'].fillna('no_keyword', inplace=True)
train['location'].fillna('no_location', inplace=True)
train.isnull().sum().sum()
train['location'].value_counts()[:10]
train['location'].value_counts()[:20].plot(kind='bar')
train['keyword'].value_counts()[:10]
train['keyword'].value_counts()[:20].plot(kind='bar')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english'))
import re
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, train.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', train['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
corpus[:5]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(corpus).toarray()
X[:10]
from sklearn.model_selection import train_test_split
y = train.iloc[:, -1].values
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()