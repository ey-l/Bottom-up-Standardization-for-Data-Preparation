import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings(action='ignore')
tweets = pd.read_csv('data/input/nlp-getting-started/train.csv')
tweets_test = pd.read_csv('data/input/nlp-getting-started/test.csv')
tweets.head()
tweets['keyword'].fillna(' ', inplace=True)
tweets['new_text'] = tweets['keyword'] + tweets['text']
tweets_df = tweets.drop(['keyword', 'text', 'location', 'id'], axis=1)
X = tweets_df.drop(['target'], axis=1)
y = tweets_df['target']
import re, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def clean_tweets(sentences):
    cleaned_sent = []
    for i in range(len(sentences)):
        cs = re.sub('[^a-zA-Z]', ' ', sentences['new_text'][i])
        cs = cs.lower()
        cs = cs.split()
        cs = [stemmer.stem(word) for word in cs if not word in stopwords.words('english')]
        cs = ' '.join(cs)
        cleaned_sent.append(cs)
    return cleaned_sent
cleaned_tweets = clean_tweets(X)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000, ngram_range=(1, 4))
X = cv.fit_transform(cleaned_tweets).toarray()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
feature_names = cv.get_feature_names()
final_train = pd.DataFrame(X_train, columns=feature_names)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.1)