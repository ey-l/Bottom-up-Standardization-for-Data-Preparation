import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
tr = _input1.copy()
tr.shape
tr.head()
tr = tr.drop('id', axis=1, inplace=False)
test_id = _input0['id']
duplicates_record = tr[tr.duplicated(['text'], keep=False)]
duplicates_record
duplicates_record.shape
tr = tr.drop_duplicates(subset=['text', 'target'], keep='first', inplace=False, ignore_index=True)
duplicates_record = tr[tr.duplicated(['text'], keep=False)]
duplicates_record.head(6)
tr = tr.drop_duplicates(subset=['text'], keep=False, inplace=False, ignore_index=True)
tr.isna().sum()
tr['keyword'].value_counts()
tr['location'].nunique()
tr[~tr['location'].isna()]
tr['target'].value_counts()
sns.countplot(tr['target'])
Y = tr['target']
tr = tr.drop('target', axis=1, inplace=False)
tr.shape
_input0.shape
_input0.head()
tr = pd.concat([tr, _input0], axis=0)
tr.shape
alpha = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()
tr['keyword'].nunique()
tr['keyword'] = np.where(tr['keyword'].isna(), 'missing', tr['keyword'])
tr['keyword'].unique()

def cleanKeyword(text):
    text = text.lower()
    text = text.replace('%20', ' ')
    text = ' '.join([ps.stem(word) for word in text.split(' ')])
    return text
tr['clean_keyword'] = tr['keyword'].apply(cleanKeyword)
tr['clean_keyword'].unique()
tr['clean_keyword'].nunique()
stopWords = stopwords.words('english')
punct = string.punctuation
punct

def cleanText(text):
    text = text.lower()
    text = ''.join([char for char in text if char in alpha])
    text = ' '.join([ps.stem(word) for word in text.split(' ') if (word not in stopWords) & (len(word) > 1)])
    return text
tr['text_clean'] = tr['text'].apply(cleanText)
tr.head()
tr['clean_tweet'] = tr['text_clean'] + ' ' + tr['clean_keyword']
vector = TfidfVectorizer(sublinear_tf=True, max_features=2700)
X = vector.fit_transform(tr['clean_tweet'].values)
X_col = vector.get_feature_names()
df = pd.DataFrame.sparse.from_spmatrix(X, columns=X_col)
df.head()
df.shape
_input0 = df.iloc[7485:]
_input0 = _input0.reset_index(drop=True, inplace=False)
_input0.head()
train_df = df.iloc[:7485]
_input0.shape
cv = KFold(n_splits=10, random_state=1, shuffle=True)
model = LogisticRegression()
scores = cross_val_score(model, train_df, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: ', scores.mean())