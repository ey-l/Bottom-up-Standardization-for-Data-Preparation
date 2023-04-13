import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 200)
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.info()
_input1.columns
f = ['keyword', 'location', 'target']
plt.figure(figsize=(5, 5))
ax = sn.countplot(x='target', data=_input1)
for p in ax.patches:
    ax.annotate(round(100 * p.get_height() / len(_input1), 2), (p.get_x() + 0.3, p.get_height() + 2))
print(_input1.target.value_counts())
print('Total Number:', sum(_input1.target.value_counts()))
sn.set(style='darkgrid')
plt.figure(figsize=(25, 75))
sn.countplot(y='keyword', data=_input1, order=_input1['keyword'].value_counts().index)
print(_input1.keyword.value_counts())
print(_input1.location.value_counts())
_input1 = _input1[['text', 'target']]
_input1.isnull().sum()
from sklearn.model_selection import train_test_split
data = _input1
X = data.text.values
y = data.target.values
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.1, random_state=101)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
indexNames = _input0[_input0['text'].isnull()].index
print(indexNames)
_input0 = _input0[['id', 'text']]
_input0.sample(5)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    s = re.sub("\\'t", ' not', s)
    s = re.sub('(@.*?)[\\s]', ' ', s)
    s = re.sub('([\\\'\\"\\.\\(\\)\\!\\?\\\\\\/\\,])', ' \\1 ', s)
    s = re.sub('[^\\w\\s\\?]', ' ', s)
    s = re.sub('([\\;\\:\\|•«\\n])', ' ', s)
    s = ' '.join([word for word in s.split() if word not in stopwords.words('english') or word in ['not', 'can']])
    s = re.sub('\\s+', ' ', s).strip()
    return s
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
model = MultinomialNB(alpha=0.1)