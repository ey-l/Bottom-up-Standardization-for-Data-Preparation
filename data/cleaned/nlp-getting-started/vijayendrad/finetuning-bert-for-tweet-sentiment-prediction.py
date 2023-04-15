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
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.head()
df.info()
df.columns
f = ['keyword', 'location', 'target']
plt.figure(figsize=(5, 5))
ax = sn.countplot(x='target', data=df)
for p in ax.patches:
    ax.annotate(round(100 * p.get_height() / len(df), 2), (p.get_x() + 0.3, p.get_height() + 2))
print(df.target.value_counts())
print('Total Number:', sum(df.target.value_counts()))
sn.set(style='darkgrid')
plt.figure(figsize=(25, 75))
sn.countplot(y='keyword', data=df, order=df['keyword'].value_counts().index)

print(df.keyword.value_counts())
print(df.location.value_counts())
df = df[['text', 'target']]
df.isnull().sum()
from sklearn.model_selection import train_test_split
data = df
X = data.text.values
y = data.target.values
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.1, random_state=101)
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
indexNames = test_data[test_data['text'].isnull()].index
print(indexNames)
test_data = test_data[['id', 'text']]
test_data.sample(5)
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