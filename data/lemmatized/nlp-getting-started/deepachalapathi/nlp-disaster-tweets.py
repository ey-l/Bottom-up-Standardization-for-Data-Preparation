import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
print(_input1.info())
_input1.isnull().sum()
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(max_features=2000, binary=True)
X = tf_idf.fit_transform(_input1['text']).toarray()
y = _input1.iloc[:, -1].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)