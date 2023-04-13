import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1 = _input1[['text', 'target']]
_input1.head()
sns.countplot(_input1['target'])
_input1['text'][0]
STOPWORDS = stopwords.words('english')

def clean_data(text):
    text = str(text).lower()
    text = re.sub('\\W+', ' ', text)
    text = re.sub('[0-9]+', ' ', text)
    text = re.sub('http\\S+', '', text)
    text = ' '.join([i for i in text.split() if i not in STOPWORDS])
    return text
ind = 0
print('Before : \n')
print(_input1.text[ind])
print('After : \n')
print(clean_data(_input1['text'][ind]))
x = _input1['text'].apply(lambda x: clean_data(x))
x = np.array(x.values)
x = x.tolist()
y = np.array(_input1['target'].values)
y.shape
tf = TfidfVectorizer()
x = tf.fit_transform(x)
x = x.toarray()
print('Matrix : ')
print(x)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.15, random_state=42)
model = LogisticRegression()