import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib as mlt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
test.head()
train

def cleaningfun(df):
    corpus = []
    emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    for sent in list(df['text']):
        sent = re.sub('\\w*\\@\\w*', '', sent)
        sent = emoji_pattern.sub('', sent)
        sent = re.sub('(?i)\\b((?:https?://|www\\d{0,3}[.]|[a-z0-9.\\-]+[.][a-z]{2,4}/)(?:[^\\s()<>]+|\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\))+(?:\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\)|[^\\s`!()\\[\\]{};:\'".,<>?«»“”‘’]))', ' ', sent)
        sent = re.sub('<.*?>', '', sent)
        sent = re.sub('[^a-zA-Z]+', ' ', sent)
        sent = sent.lower()
        sent = sent.split()
        sent = [ps.stem(word) for word in sent if not word in stopwords.words('english')]
        sent = ' '.join(sent)
        corpus.append(sent)
    return corpus
corpus_train = cleaningfun(train)
corpus_test = cleaningfun(test)
hidden_train = pd.DataFrame(corpus_train, index=train.id, columns=['text'])
hidden_test = pd.DataFrame(corpus_test, index=test.id, columns=['text'])
hidden_train
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_train = TfidfVectorizer(max_features=2500)
Mat_train = vectorizer_train.fit_transform(corpus_train)
tfidf_tokens_train = vectorizer_train.get_feature_names()
Mat_train.shape
Mat_test = vectorizer_train.transform(corpus_test)
Mat_test.shape
X = pd.DataFrame(data=Mat_train.toarray(), index=train.id, columns=tfidf_tokens_train)
X
test_data = pd.DataFrame(data=Mat_test.toarray(), index=test.id, columns=tfidf_tokens_train)
test_data.head()
y = train['target']
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=20)
X_train.info()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()