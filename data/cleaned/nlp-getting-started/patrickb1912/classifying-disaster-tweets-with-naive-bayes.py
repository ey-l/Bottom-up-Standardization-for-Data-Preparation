import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.head()
df['keyword'].unique()
df['location'].unique()
df = df.drop(['id', 'keyword', 'location'], axis=1)
df
X = df['text']
y = df['target']
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.25)
X_train.head()
tokenizer = RegexpTokenizer('\\w+')
en_stop = set(stopwords.words('english'))
ps = PorterStemmer()

def getStemmedTweet(tweet):
    """
        This function takes the tweet string and then performs the preprocessing steps on it
        to return the cleaned tweet which will be more effective in predictions later made by the 
        classifier.
    """
    tweet = tweet.lower()
    tokens = tokenizer.tokenize(tweet)
    new_tokens = [token for token in tokens if token not in en_stop]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    cleaned_review = ' '.join(stemmed_tokens)
    return cleaned_review
rand_num = 34
print('Review ===> ', X_train[rand_num])
print('Preprocessed Review ===>', getStemmedTweet(X_train[rand_num]))
X_cleaned = X_train.apply(getStemmedTweet)
Xval_cleaned = X_val.apply(getStemmedTweet)
cv = CountVectorizer()
X_vec = cv.fit_transform(X_cleaned).toarray()
Xval_vec = cv.transform(Xval_cleaned).toarray()
print(X_vec.shape)
print(Xval_vec.shape)
mnb = MultinomialNB()