import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
ss = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')



y = train['target']
train = train.drop(['target'], axis=1)
X = pd.concat([train, test], axis=0, ignore_index=True)
X = X.drop(['id'], axis=1)

val1 = pd.unique(X['keyword'])
print(val1)
print(len(val1))
val2 = pd.unique(X['location'])
print(val2)
print(len(val2))
X['keyword'] = X['keyword'].fillna('No keyword.')
X = X.drop(['location'], axis=1)
X['text'] = X['text'].str.lower()
X['keyword'] = LabelEncoder().fit_transform(X['keyword'])

print(y.mean())