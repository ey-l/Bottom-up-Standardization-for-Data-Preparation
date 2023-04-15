import numpy as np
import pandas as pd
import nltk
tweets = pd.read_csv('data/input/nlp-getting-started/train.csv')
pd.pandas.set_option('display.max_rows', None)
tweets
tweets['keyword'].unique()
tweets['location'].unique()
len(tweets['location'].unique())
len(tweets['keyword'].unique())
tweets.columns
data = pd.read_csv('data/input/nlp-getting-started/train.csv', usecols=['text', 'target'])
data['target'].value_counts()
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
ls = WordNetLemmatizer()
corpus = []
for i in range(0, len(data)):
    tweets = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    tweets = tweets.lower()
    tweets = tweets.split()
    tweets = [ls.lemmatize(word) for word in tweets if word not in stopwords.words('english')]
    tweets = ' '.join(tweets)
    corpus.append(tweets)
corpus
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
X
y = data['target']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.naive_bayes import MultinomialNB