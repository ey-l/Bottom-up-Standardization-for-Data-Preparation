import numpy as np
import pandas as pd
import re
import nltk
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head(15)
_input0.head()
_input1.shape
_input1.isna().sum()
_input1.info()
_input1.describe()
_input1.isna().sum()
_input1['keyword'].unique()
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 7613):
    text = re.sub('[^a-zA-Z]', ' ', _input1['text'][i])
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in text if not word in set(all_stopwords)]
    review = ' '.join(text)
    corpus.append(text)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=7613)
X = cv.fit_transform(corpus).toarray()
y = _input1.iloc[:, [4]].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()