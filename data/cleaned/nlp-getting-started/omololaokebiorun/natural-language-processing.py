import pandas as pd
import seaborn as sns
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
test.isna().sum()
train.head(5)
test.head(5)
train.info()
test.info()
train.drop(columns=['id', 'keyword', 'location'], axis=1, inplace=True)
test.drop(columns=['id', 'keyword', 'location'], axis=1, inplace=True)
train.isna().sum()
train.duplicated().sum()
train.drop_duplicates(inplace=True)
train.target.value_counts(normalize=True)
sns.countplot(x='target', data=train, palette='winter')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
en_stopwords = stopwords.words('english')

def Preprocess(tweets):
    tweet = re.sub('[^A-Za-z1-9 ]', '', tweets)
    lower_case = tweet.lower()
    tokens = word_tokenize(lower_case)
    clean_tweet = []
    for token in tokens:
        if token not in en_stopwords:
            clean_tweet.append(lemma.lemmatize(token))
    return ' '.join(clean_tweet)
train.text = train.text.apply(Preprocess)
train.text
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tweet_matrix = vectorizer.fit_transform(train['text'])
X = tweet_matrix
y = train.target
X.shape
y.shape
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
from sklearn.model_selection import cross_val_score
cross_val_score(logistic, X, y, cv=5).mean()