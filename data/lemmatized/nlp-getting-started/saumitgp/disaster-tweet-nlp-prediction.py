import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.shape
_input1.info()
_input1.sample(n=20)
_input1.groupby('target').count()
_input1.isnull().sum()
_input1.keyword = _input1.keyword.fillna('')
_input1.location = _input1.location.fillna('')
_input1.isnull().sum()
(x_train, x_test, y_train, y_test) = train_test_split(_input1.text, _input1.target, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = feature_extraction.text.TfidfVectorizer()
train_vectors = vectorizer.fit_transform(_input1['text'])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, _input1['target'], cv=3, scoring='f1')
scores