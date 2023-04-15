import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
TRAIN_PATH = 'data/input/nlp-getting-started/train.csv'
TEST_PATH = 'data/input/nlp-getting-started/test.csv'
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
for i in range(len(train)):
    train.text[i] = train.text[i].lower()
for i in range(len(test)):
    test.text[i] = test.text[i].lower()
tokenizer = nltk.tokenize.TreebankWordTokenizer()
stop_words = stopwords.words('english')
for i in range(len(train)):
    tokens = tokenizer.tokenize(train.text[i])
    review = [i for i in tokens if not i in stop_words]
    train.text[i] = ' '.join(review)
for i in range(len(test)):
    tokens = tokenizer.tokenize(test.text[i])
    review = [i for i in tokens if not i in stop_words]
    test.text[i] = ' '.join(review)
lemmatizer = nltk.stem.WordNetLemmatizer()
for i in range(len(train)):
    tokens = tokenizer.tokenize(train.text[i])
    train.text[i] = ' '.join((lemmatizer.lemmatize(token) for token in tokens))
for i in range(len(test)):
    tokens = tokenizer.tokenize(test.text[i])
    test.text[i] = ' '.join((lemmatizer.lemmatize(token) for token in tokens))
train.head()
train = train.drop(labels=['keyword', 'location'], axis=1)
test = test.drop(labels=['keyword', 'location'], axis=1)
tfidf = TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1, 2))
fit_tfidf = tfidf.fit_transform(train.text).toarray()
train_data = pd.DataFrame(fit_tfidf, columns=tfidf.get_feature_names())
test_data = pd.DataFrame(tfidf.transform(test.text).toarray(), columns=tfidf.get_feature_names())
(x_train, x_val, y_train, y_val) = train_test_split(train_data, train.target, test_size=0.2, random_state=47)
logreg = LogisticRegression()