import numpy as np
import pandas as pd
import re
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1
_input1 = _input1.drop(['keyword', 'location'], axis=1)
_input1.isnull().sum()
import re
import nltk
nltk.download('all')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
corpus = []
for i in range(0, 7613):
    review_train = re.sub('[^a-zA-Z]', ' ', _input1['text'][i])
    review_train = re.sub('http\\S+', '', review_train)
    review_train = re.sub('#([^\\s]+)', '\\1', review_train)
    review_train = re.sub('[\\s]+', ' ', review_train)
    review_train = review_train.lower()
    review_train = review_train.split()
    lemmatizer = WordNetLemmatizer()
    stopwords1 = stopwords.words('english')
    review_train = [lemmatizer.lemmatize(word) for word in review_train if not word in stopwords1]
    review_train = ' '.join(review_train)
    corpus.append(review_train)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
corpus_test = []
for i in range(0, 3263):
    review = re.sub('[^a-zA-Z]', ' ', _input0['text'][i])
    review = re.sub('http\\S+', '', review)
    review = re.sub('#([^\\s]+)', '\\1', review)
    review = re.sub('[\\s]+', ' ', review)
    review = review.lower()
    review = review.split()
    lemmatizer_2 = WordNetLemmatizer()
    stopwords_2 = stopwords.words('english')
    review = [lemmatizer_2.lemmatize(word) for word in review if not word in stopwords_2]
    review = ' '.join(review)
    corpus_test.append(review)
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=2, max_features=None, analyzer='word', ngram_range=(1, 3))
X_vect = vect.fit_transform(corpus)
y_vect = _input1['target']
X_vect.shape
y_vect.shape
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X_vect, y_vect, test_size=0.2, random_state=1)
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression(penalty='l2', C=3, max_iter=550)