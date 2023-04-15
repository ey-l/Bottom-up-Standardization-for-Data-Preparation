import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head(10)
data.shape
data['Outcome'].value_counts()
data['Outcome'].value_counts(normalize=True).apply(lambda x: x * 100)
y = data['Outcome']
y.shape
X = data.copy()
X.pop('Outcome')
X.shape
X.head()
(X.shape, y.shape)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
(X_train.shape, X_test.shape)
print(X_train)
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
print(X_train)
from sklearn.svm import SVC
SVM_classifier = SVC(kernel='linear', random_state=0)