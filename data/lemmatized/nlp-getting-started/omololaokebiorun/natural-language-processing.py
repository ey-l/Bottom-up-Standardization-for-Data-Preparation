import pandas as pd
import seaborn as sns
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0.isna().sum()
_input1.head(5)
_input0.head(5)
_input1.info()
_input0.info()
_input1 = _input1.drop(columns=['id', 'keyword', 'location'], axis=1, inplace=False)
_input0 = _input0.drop(columns=['id', 'keyword', 'location'], axis=1, inplace=False)
_input1.isna().sum()
_input1.duplicated().sum()
_input1 = _input1.drop_duplicates(inplace=False)
_input1.target.value_counts(normalize=True)
sns.countplot(x='target', data=_input1, palette='winter')
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
_input1.text = _input1.text.apply(Preprocess)
_input1.text
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tweet_matrix = vectorizer.fit_transform(_input1['text'])
X = tweet_matrix
y = _input1.target
X.shape
y.shape
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
from sklearn.model_selection import cross_val_score
cross_val_score(logistic, X, y, cv=5).mean()