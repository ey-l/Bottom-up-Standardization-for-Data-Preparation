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
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
train = train[['text', 'target']]
train.head()
sns.countplot(train['target'])
train['text'][0]
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
print(train.text[ind])
print('After : \n')
print(clean_data(train['text'][ind]))
x = train['text'].apply(lambda x: clean_data(x))
x = np.array(x.values)
x = x.tolist()
y = np.array(train['target'].values)
y.shape
tf = TfidfVectorizer()
x = tf.fit_transform(x)
x = x.toarray()
print('Matrix : ')
print(x)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.15, random_state=42)
model = LogisticRegression()