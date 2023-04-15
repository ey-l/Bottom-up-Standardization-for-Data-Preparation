import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
df_test = pd.read_csv('data/input/nlp-getting-started/test.csv')
df_train
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
cleaned_tweet = []
for i in range(0, len(df_train)):
    tweet = re.sub('[^a-zA-Z]', ' ', df_train['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    cleaned_tweet.append(tweet)
cleaned_tweet_test = []
for i in range(0, len(df_test)):
    tweet = re.sub('[^a-zA-Z]', ' ', df_test['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    cleaned_tweet_test.append(tweet)
print(cleaned_tweet_test)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(cleaned_tweet)
test_vectors = cv.transform(cleaned_tweet_test)
y = df_train['target']
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = linear_model.RidgeClassifier()