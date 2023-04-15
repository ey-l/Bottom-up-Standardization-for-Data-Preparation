import numpy as np
import pandas as pd
import re
import nltk
data_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
data_test = pd.read_csv('data/input/nlp-getting-started/test.csv')
data_train.head(15)
data_test.head()
data_train.shape
data_train.isna().sum()
data_train.info()
data_train.describe()
data_train.isna().sum()
data_train['keyword'].unique()
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 7613):
    text = re.sub('[^a-zA-Z]', ' ', data_train['text'][i])
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in text if not word in set(all_stopwords)]
    review = ' '.join(text)
    corpus.append(text)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=7613)
X = cv.fit_transform(corpus).toarray()
y = data_train.iloc[:, [4]].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()