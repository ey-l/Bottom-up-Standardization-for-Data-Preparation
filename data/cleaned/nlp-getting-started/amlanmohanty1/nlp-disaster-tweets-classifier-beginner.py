import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
train_df.head()
train_df.describe()
train_df.isna().sum()
train_df['target'].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', figsize=(6, 6))
import seaborn as sns
sns.countplot(x='target', data=train_df)
train_df[train_df.keyword != 'NaN'].value_counts()
train_df = train_df.drop(['location', 'keyword', 'id'], axis=1)
train_df.head()
train_df.isna().sum()
train_df.info()
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
test_df
test_df.isna().sum()
test_df = test_df.drop(['location', 'keyword'], axis=1)
test_df.head()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
train_df.text = train_df.text.apply(lambda x: x.lower())
train_df

import contractions

def con(data):
    expand = contractions.fix(data)
    return expand
train_df.text = train_df.text.apply(con)
train_df['text'][0]
import re

def remove_sp(data):
    pattern = '[^A-Za-z0-9\\s]'
    data = re.sub(pattern, '', data)
    return data
train_df.text = train_df.text.apply(remove_sp)
train_df.text[0]
nltk.download('stopwords')
stopword_list = stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
train_df.text = train_df.text.apply(lambda x: ' '.join((x for x in x.split() if x not in stopword_list)))
train_df['text'][5]
nltk.download('punkt')
train_df['text'] = train_df.text.apply(word_tokenize)
train_df['text'][0]
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
train_df['text'] = train_df.text.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
train_df.text
train_df.text = train_df.text.astype(str)
train_df.head()
X = train_df.text
Y = train_df.target
X_test = test_df.text
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
x_train_tfidf = tfidf.fit_transform(X)
np.random.seed(42)
from sklearn.svm import SVC
svc_clf = SVC()