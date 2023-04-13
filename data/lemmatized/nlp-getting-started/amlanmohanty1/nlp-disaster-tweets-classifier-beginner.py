import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.describe()
_input1.isna().sum()
_input1['target'].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', figsize=(6, 6))
import seaborn as sns
sns.countplot(x='target', data=_input1)
_input1[_input1.keyword != 'NaN'].value_counts()
_input1 = _input1.drop(['location', 'keyword', 'id'], axis=1)
_input1.head()
_input1.isna().sum()
_input1.info()
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0
_input0.isna().sum()
_input0 = _input0.drop(['location', 'keyword'], axis=1)
_input0.head()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
_input1.text = _input1.text.apply(lambda x: x.lower())
_input1
import contractions

def con(data):
    expand = contractions.fix(data)
    return expand
_input1.text = _input1.text.apply(con)
_input1['text'][0]
import re

def remove_sp(data):
    pattern = '[^A-Za-z0-9\\s]'
    data = re.sub(pattern, '', data)
    return data
_input1.text = _input1.text.apply(remove_sp)
_input1.text[0]
nltk.download('stopwords')
stopword_list = stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
_input1.text = _input1.text.apply(lambda x: ' '.join((x for x in x.split() if x not in stopword_list)))
_input1['text'][5]
nltk.download('punkt')
_input1['text'] = _input1.text.apply(word_tokenize)
_input1['text'][0]
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
_input1['text'] = _input1.text.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
_input1.text
_input1.text = _input1.text.astype(str)
_input1.head()
X = _input1.text
Y = _input1.target
X_test = _input0.text
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
x_train_tfidf = tfidf.fit_transform(X)
np.random.seed(42)
from sklearn.svm import SVC
svc_clf = SVC()